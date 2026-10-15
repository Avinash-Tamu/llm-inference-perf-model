"""Shared helpers for the v0.2 energy/DVFS analysis (reads cached LLMCompass runs)."""
import json, sys
from dataclasses import replace

sys.path.insert(0, ".")
from software_model.llama import LLAMA3_70B, LLAMA1_65B, LlamaDecoderLayerTP
from software_model.utils import Tensor
from hardware_model.energy_model import (EnergyParams, llama_layer_counts, lm_head_counts,
                                         add_counts, calibrate_e_mac)
from hardware_model.compute_module import overhead_dict

TP = 4
FREQS = [0.60e9, 0.75e9, 0.90e9, 1.05e9, 1.20e9, 1.305e9, 1.41e9]
WORKLOADS = {
    "decode_b1": (LLAMA3_70B, "decode", 1, 2048),
    "decode_b64": (LLAMA3_70B, "decode", 64, 2048),
    "prefill_2k": (LLAMA3_70B, "prefill", 1, 2048),
    "wtw_65b": (LLAMA1_65B, "decode", 64, 384),
}
FREQS_FOR = {"prefill_2k": [0.60e9, 0.90e9, 1.20e9, 1.41e9]}
CACHE = "myexp/results/dvfs_points.json"


def load_cache():
    return json.load(open(CACHE))


def params_for(cache, **overrides) -> EnergyParams:
    """EnergyParams with e_mac calibrated (400 W on an 8192^3 GEMM at f0) after overrides."""
    g = cache["gemm"]
    n = g["n"]
    return calibrate_e_mac(replace(EnergyParams(), **overrides), n ** 3, 3 * n * n * 2, g["latency_s"])


def kernel_overhead(cfg, mode):
    """LLMCompass's fixed A100 per-kernel overheads summed over one forward pass (s)."""
    o = overhead_dict["A100"]
    per_layer = 9 * o.matmul + o.softmax + 2 * o.layernorm + o.gelu
    return cfg.n_layers * per_layer + o.matmul


def workload_counts(name, l1_tile=128):
    cfg, mode, b, n = WORKLOADS[name]
    L = LlamaDecoderLayerTP(cfg, TP)
    if mode == "decode":
        L(Tensor([b, 1, cfg.d_model]), "decode", kv_len=n - 1)
    else:
        L(Tensor([b, n, cfg.d_model]), "prefill")
    return add_counts(llama_layer_counts(L, l1_tile), lm_head_counts(cfg, b, TP), scale=[cfg.n_layers, 1])


def curve(cache, name, params, overhead_clock_frac=0.0):
    """List of dicts over the clock grid: latency, W/GPU, J/token, energy shares.
    overhead_clock_frac: fraction of LLMCompass's fixed per-kernel overhead treated as
    GPU-side work that stretches with 1/f (0 = all CPU-side launch latency)."""
    cfg, mode, b, n = WORKLOADS[name]
    counts = workload_counts(name, params.l1_tile)
    ovh = kernel_overhead(cfg, mode)
    tokens = b if mode == "decode" else b * n
    out = []
    for f in FREQS_FOR.get(name, FREQS):
        r = cache[f"{name},{f:.0f}"]
        t = cfg.n_layers * r["layer_s"] + r["head_s"]
        t += overhead_clock_frac * ovh * (params.f0 / f - 1)
        e = params.energy(counts, t, f)
        tot = sum(e.values())
        row = {"workload": name, "f_GHz": round(f / 1e9, 3), "V_norm": round(params.voltage(f), 3),
               "latency_ms": t * 1e3, "power_W_per_gpu": tot / t, "J_per_token": TP * tot / tokens}
        row.update({f"share_{k}": v / tot for k, v in e.items()})
        out.append(row)
    return out
