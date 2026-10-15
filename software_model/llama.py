"""
Llama-style decoder layer for LLMCompass: grouped-query attention (GQA) + SwiGLU FFN.

Added on top of LLMCompass (Zhang et al., ISCA'24). The upstream TransformerBlock*
classes model GPT-3 (multi-head attention, 4d GeLU FFN). This file adds:

  * n_kv_heads: number of K/V heads (GQA). n_kv_heads == n_heads gives plain MHA,
    n_kv_heads == 1 gives MQA. Each K/V head serves a group of g = n_heads/n_kv_heads
    query heads, so attention is modeled as a batched matmul of shape
    [b * n_kv, g * s_q, d_h] x [b * n_kv, d_h, s_kv]: the query group becomes the
    matmul M dimension, which is where the GQA arithmetic-intensity gain comes from.
  * SwiGLU FFN: gate and up projections [d, d_ff], SiLU, elementwise multiply, down [d_ff, d].
  * RMSNorm, modeled with the LayerNorm operator (slightly pessimistic).
  * Tensor parallelism across `device_count` devices (heads and d_ff split; two
    all-reduces per layer), same scheme as upstream.

Operators the upstream code does not simulate (reshape/transpose/concat) are not
counted here either, to stay comparable with the paper's validated numbers.
"""
from dataclasses import dataclass
from typing import Dict

from software_model.operators import Operator
from software_model.matmul import Matmul, BatchedMatmul
from software_model.softmax import Softmax
from software_model.layernorm import LayerNorm
from software_model.gelu import GeLU
from software_model.communication_primitives import AllReduceMultiPCB
from software_model.utils import Tensor, DataType, data_type_dict
from hardware_model.system import System


@dataclass(frozen=True)
class LlamaConfig:
    name: str
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    n_layers: int
    vocab: int

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def kv_bytes_per_token(self, word_size: int = 2) -> int:
        """KV cache bytes per token for the whole model (all layers, all devices)."""
        return 2 * self.n_layers * self.n_kv_heads * self.d_head * word_size

    def param_count(self) -> int:
        d, dh = self.d_model, self.d_head
        attn = d * self.n_heads * dh + 2 * d * self.n_kv_heads * dh + self.n_heads * dh * d
        ffn = 3 * d * self.d_ff
        norms = 2 * d
        per_layer = attn + ffn + norms
        return self.n_layers * per_layer + 2 * self.vocab * d + d  # embed + lm_head + final norm


# Public config from meta-llama/Meta-Llama-3-70B config.json
LLAMA3_70B = LlamaConfig("Llama-3-70B", 8192, 64, 8, 28672, 80, 128256)
# Public config from meta-llama/Llama-2-70b-hf config.json. Same layer shapes as
# Llama-3-70B, smaller vocab. Used to validate against published Llama-2-70B benchmarks.
LLAMA2_70B = LlamaConfig("Llama-2-70B", 8192, 64, 8, 28672, 80, 32000)
# LLaMA-1 65B (MHA, d_ff 22016), used to compare against published A100 power-capping data.
LLAMA1_65B = LlamaConfig("LLaMA-65B", 8192, 64, 64, 22016, 80, 32000)
# Same model with MHA (n_kv_heads = n_heads) as a counterfactual baseline
LLAMA3_70B_MHA = LlamaConfig("Llama-3-70B (MHA baseline)", 8192, 64, 64, 28672, 80, 128256)


class ElementwiseMul(Operator):
    """out = a * b, purely memory bound: read 2 tensors, write 1."""

    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        assert a.shape == b.shape
        self.n = a.size
        self.flop_count = self.n
        self.io_count = 3 * self.n
        return Tensor(a.shape, self.data_type)

    def _latency(self, device) -> float:
        bw = min(
            device.io_module.bandwidth,
            device.compute_module.l2_bandwidth_per_cycle * device.compute_module.clock_freq,
        )
        io = self.io_count * self.data_type.word_size / bw
        compute = self.flop_count / device.compute_module.total_vector_flops
        return max(io, compute)

    def roofline_model(self, device) -> float:
        return self._latency(device)

    def compile_and_simulate(self, device, compile_mode=None) -> float:
        return self._latency(device)


class LlamaDecoderLayerTP(Operator):
    """One Llama decoder layer, per device, under `device_count`-way tensor parallelism.

    mode="prefill": input [b, s, d], attends over s tokens (no causal-mask savings modeled).
    mode="decode":  input [b, 1, d], attends over a KV cache of `kv_len` tokens.
    """

    MATMULS = ["q_proj", "k_proj", "v_proj", "qk", "av", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def __init__(self, cfg: LlamaConfig, device_count: int, data_type: DataType = data_type_dict["fp16"]):
        super().__init__(0, 0, 0, 0, data_type)
        assert cfg.n_heads % device_count == 0 and cfg.n_kv_heads % device_count == 0
        assert cfg.d_ff % device_count == 0
        self.cfg = cfg
        self.device_count = device_count
        d, dh, dev = cfg.d_model, cfg.d_head, device_count
        self.h = cfg.n_heads // dev
        self.kv = cfg.n_kv_heads // dev
        self.g = cfg.n_heads // cfg.n_kv_heads
        self.ff = cfg.d_ff // dev
        dt = data_type
        # per-device weights
        self.Wq = Tensor([d, self.h * dh], dt)
        self.Wk = Tensor([d, self.kv * dh], dt)
        self.Wv = Tensor([d, self.kv * dh], dt)
        self.Wo = Tensor([self.h * dh, d], dt)
        self.Wg = Tensor([d, self.ff], dt)
        self.Wu = Tensor([d, self.ff], dt)
        self.Wd = Tensor([self.ff, d], dt)
        # operators
        self.ops: Dict[str, Operator] = {
            "attn_norm": LayerNorm(dt),
            "q_proj": Matmul(dt),
            "k_proj": Matmul(dt),
            "v_proj": Matmul(dt),
            "qk": BatchedMatmul(dt),
            "softmax": Softmax(dt),
            "av": BatchedMatmul(dt),
            "o_proj": Matmul(dt),
            "allreduce_attn": AllReduceMultiPCB(dt),
            "ffn_norm": LayerNorm(dt),
            "gate_proj": Matmul(dt),
            "up_proj": Matmul(dt),
            "silu": GeLU(dt),  # stand-in: same vector-op cost class as SiLU
            "glu_mul": ElementwiseMul(dt),
            "down_proj": Matmul(dt),
            "allreduce_ffn": AllReduceMultiPCB(dt),
        }

    def __call__(self, x: Tensor, mode: str, kv_len: int = 0) -> Tensor:
        assert mode in ("prefill", "decode")
        self.mode = mode
        o, dt = self.ops, self.data_type
        b, s, d = x.shape
        dh, h, kv, g = self.cfg.d_head, self.h, self.kv, self.g
        s_kv = s if mode == "prefill" else kv_len + 1
        self.b, self.s, self.s_kv = b, s, s_kv

        x = o["attn_norm"](x)
        q = o["q_proj"](x, self.Wq)  # [b, s, h*dh]
        k = o["k_proj"](x, self.Wk)  # [b, s, kv*dh]
        v = o["v_proj"](x, self.Wv)
        # GQA: group g query heads per KV head -> M = g * s
        q_g = Tensor([b * kv, g * s, dh], dt)
        k_t = Tensor([b * kv, dh, s_kv], dt)  # full K (cache + new) for decode
        v_c = Tensor([b * kv, s_kv, dh], dt)
        scores = o["qk"](q_g, k_t)  # [b*kv, g*s, s_kv]
        o["softmax"](Tensor([b * kv * g * s, s_kv], dt))  # row-wise softmax over s_kv
        probs = Tensor(scores.shape, dt)
        o["av"](probs, v_c)  # [b*kv, g*s, dh]
        attn = Tensor([b, s, h * dh], dt)  # regroup heads (reshape/transpose not simulated)
        y = o["o_proj"](attn, self.Wo)  # [b, s, d]
        if self.device_count > 1:
            y = o["allreduce_attn"](y)
        y = o["ffn_norm"](y)
        gt = o["gate_proj"](y, self.Wg)
        up = o["up_proj"](y, self.Wu)
        gt = o["silu"](gt)
        z = o["glu_mul"](gt, up)
        out = o["down_proj"](z, self.Wd)
        if self.device_count > 1:
            out = o["allreduce_ffn"](out)

        wsz = dt.word_size
        self.weight_bytes = sum(t.size for t in [self.Wq, self.Wk, self.Wv, self.Wo, self.Wg, self.Wu, self.Wd]) * wsz
        self.kv_cache_bytes = 2 * b * kv * s_kv * dh * wsz
        return out

    # ---------- per-operator accounting (for roofline plots) ----------
    def op_stats(self) -> Dict[str, Dict[str, float]]:
        """FLOPs and DRAM bytes per operator (ideal: every operand read/written once)."""
        wsz = self.data_type.word_size
        stats = {}
        for name in self.MATMULS:
            op = self.ops[name]
            if isinstance(op, BatchedMatmul):
                flops = 2 * op.bs * op.M * op.K * op.N
                byts = op.bs * (op.M * op.K + op.K * op.N + op.M * op.N) * wsz
            else:
                flops = op.flop_count
                byts = op.io_count * wsz
            stats[name] = {"flops": flops, "bytes": byts}
        return stats

    # ---------- latency ----------
    def _run(self, system: System, how: str, compile_mode: str = "heuristic-GPU") -> Dict[str, float]:
        dev, ic, ovh = system.device, system.interconnect, system.device.compute_module.overhead
        lat: Dict[str, float] = {}
        for name, op in self.ops.items():
            if isinstance(op, AllReduceMultiPCB):
                lat[name] = op.simulate(ic) if self.device_count > 1 else 0.0
                continue
            if how == "roofline":
                t = op.roofline_model(dev)
            else:
                t = op.compile_and_simulate(dev, compile_mode)
            if isinstance(op, (Matmul, BatchedMatmul)):
                t += ovh.matmul
            elif isinstance(op, Softmax):
                t += ovh.softmax
            elif isinstance(op, LayerNorm):
                t += ovh.layernorm
            elif isinstance(op, GeLU):
                t += ovh.gelu
            lat[name] = t
        self.breakdown = lat
        self.latency = sum(lat.values())
        return lat

    def roofline_model(self, system: System) -> float:
        self._run(system, "roofline")
        return self.latency

    def compile_and_simulate(self, system: System, compile_mode: str = "heuristic-GPU") -> float:
        self._run(system, "simulate", compile_mode)
        return self.latency


class LMHeadTP(Operator):
    """Final vocab projection, split across devices along the vocab dimension."""

    def __init__(self, cfg: LlamaConfig, device_count: int, data_type: DataType = data_type_dict["fp16"]):
        super().__init__(0, 0, 0, 0, data_type)
        self.W = Tensor([cfg.d_model, cfg.vocab // device_count], data_type)
        self.mm = Matmul(data_type)

    def __call__(self, x: Tensor) -> Tensor:
        b, s, d = x.shape
        return self.mm(Tensor([b, 1, d], self.data_type), self.W)  # only the last token's logits

    def compile_and_simulate(self, system: System, compile_mode: str = "heuristic-GPU") -> float:
        return self.mm.compile_and_simulate(system.device, compile_mode) + system.device.compute_module.overhead.matmul

    def roofline_model(self, system: System) -> float:
        return self.mm.roofline_model(system.device) + system.device.compute_module.overhead.matmul
