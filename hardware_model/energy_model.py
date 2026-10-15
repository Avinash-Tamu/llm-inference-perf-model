"""
Per-component energy + DVFS model for LLMCompass (v0.2 extension).

Energy of one simulated interval on one device:

    E = MACs * e_mac(V) + vector_FLOPs * e_vec(V) + L2_bytes * e_l2(V)        # core-domain dynamic
      + HBM_bytes * e_mempath(V)                                             # HBM-side on-chip path
      + HBM_bytes * e_hbm + link_bytes * e_link                              # fixed-voltage I/O domains
      + [P_leak(V) + P_clk(f, V) + P_fixed] * latency                        # time-proportional power

DVFS: the core clock f and core voltage V(f) move together; HBM and NVLink stay at
their own fixed rails. Dynamic core energy scales with (V/V0)^2, clock-tree power with
f * (V/V0)^2, leakage roughly with (V/V0)^3. Latency comes from the LLMCompass
simulator run at clock f (compute cycles / f; DRAM time is f-independent), so
memory-bound phases barely slow down when f drops while their core energy falls.

All parameters are public-literature values or explicit assumptions (see PARAM_NOTES).
The V-f curve is a generic 7 nm-class assumption, NOT vendor data.
"""
from dataclasses import dataclass, field, replace
from typing import Dict


@dataclass(frozen=True)
class EnergyParams:
    f0: float = 1.41e9          # nominal (max boost) core clock, Hz  [A100 SXM spec]
    # V-f curve, normalized to V0 at f0: linear between (f_min, v_min) and (f0, 1.0), floor below f_min
    f_min: float = 0.60e9
    v_min: float = 0.65
    # dynamic energies at V0
    e_mac_pj: float = 0.0       # FP16 tensor MAC incl. register/operand delivery; set by calibrate()
    e_vec_pj: float = 2.0       # FP32 vector FLOP incl. register traffic  [assumption]
    e_l2_pj_per_byte: float = 5.0    # L2 / global-buffer access + on-chip wire, ~0.6 pJ/bit  [assumption]
    e_hbm_pj_per_byte: float = 31.2  # HBM2 ~3.9 pJ/bit  [O'Connor et al., MICRO'17]
    e_link_pj_per_byte: float = 10.0  # NVLink3 SerDes + switch, ~1.25 pJ/bit  [assumption]
    # HBM-side on-chip data path per byte moved to/from HBM (memory controller, PHY digital,
    # crossbar, L2 fill). Core voltage domain. 0 in the as-built model; fitted in the
    # calibrated model against published A100 power-capping data (myexp/compare_powercap.py).
    e_mempath_pj_per_byte: float = 0.0
    l1_tile: int = 128          # L1/SMEM tile edge used to estimate L2->L1 operand re-reads
    # time-proportional power per device at V0 / f0
    p_leak0_w: float = 40.0     # core leakage, scales ~ (V/V0)^3  [assumption]
    p_clk0_w: float = 30.0      # clock tree + always-on control, scales f * (V/V0)^2  [assumption]
    p_fixed_w: float = 35.0     # HBM background/refresh + PHY + board misc, f-independent  [assumption]

    # ---- DVFS ----
    def voltage(self, f: float) -> float:
        """Normalized core voltage V/V0 at clock f (generic assumption)."""
        if f <= self.f_min:
            return self.v_min
        return self.v_min + (1.0 - self.v_min) * (f - self.f_min) / (self.f0 - self.f_min)

    def static_power(self, f: float) -> float:
        v = self.voltage(f)
        return self.p_leak0_w * v ** 3 + self.p_clk0_w * (f / self.f0) * v ** 2 + self.p_fixed_w

    def energy(self, counts: Dict[str, float], latency_s: float, f: float) -> Dict[str, float]:
        """Joules by component for one device over an interval of `latency_s` at clock f."""
        v2 = self.voltage(f) ** 2
        pj = 1e-12
        return {
            "compute": counts["macs"] * self.e_mac_pj * v2 * pj + counts["vec_flops"] * self.e_vec_pj * v2 * pj,
            "sram": counts["l2_bytes"] * self.e_l2_pj_per_byte * v2 * pj,
            "hbm": counts["hbm_bytes"] * self.e_hbm_pj_per_byte * pj,
            "mempath": counts["hbm_bytes"] * self.e_mempath_pj_per_byte * v2 * pj,
            "link": counts["link_bytes"] * self.e_link_pj_per_byte * pj,
            "static": self.static_power(f) * latency_s,
        }


PARAM_NOTES = {
    "e_hbm_pj_per_byte": "HBM2 access energy ~3.9 pJ/bit (O'Connor et al., 'Fine-Grained DRAM', MICRO 2017).",
    "e_mac_pj": "Calibrated so a large compute-bound FP16 GEMM at f0 draws the A100 SXM TDP (400 W).",
    "e_mempath_pj_per_byte": "Fitted: uncapped LLaMA-65B decode power = 243 W/GPU implied by Samsi et al. (HPEC 2023).",
    "V-f curve": "Generic 7 nm-class linear V-f with a floor; no vendor V-f data is used.",
    "others": "Order-of-magnitude assumptions; README reports sensitivity to them.",
}


# ---------------------------------------------------------------------------
# Analytic event counts for one Llama decoder layer on one device (TP-sharded),
# matching the operator graph in software_model/llama.py.
# ---------------------------------------------------------------------------
def llama_layer_counts(layer, l1_tile: int = 128) -> Dict[str, float]:
    cfg, dev, w = layer.cfg, layer.device_count, layer.data_type.word_size
    b, s, skv = layer.b, layer.s, layer.s_kv
    d, dh, h, kv, g, ff = cfg.d_model, cfg.d_head, layer.h, layer.kv, layer.g, layer.ff
    T = b * s  # new tokens this step
    macs = vec = hbm = 0.0

    def mm(M, K, N, bs=1, weight=True):
        nonlocal macs, hbm
        macs += bs * M * K * N
        hbm += bs * (M * K + K * N + M * N) * w

    mm(T, d, h * dh); mm(T, d, kv * dh); mm(T, d, kv * dh)       # q, k, v
    mm(g * s, dh, skv, bs=b * kv); mm(g * s, skv, dh, bs=b * kv)  # qk, av (KV cache read here)
    mm(T, h * dh, d)                                               # o_proj
    mm(T, d, ff); mm(T, d, ff); mm(T, ff, d)                       # gate, up, down
    scores = b * kv * g * s * skv
    vec += 2 * 7 * T * d + 10 * scores + 8 * T * ff + T * ff       # 2 norms, softmax, silu, glu mul
    hbm += 2 * 2 * T * d * w + 2 * scores * w + 2 * T * ff * w + 3 * T * ff * w
    link = 2 * (2 * (dev - 1) / dev * T * d * w) if dev > 1 else 0.0  # two ring all-reduces
    l2 = 2 * hbm + 2 * macs * w / l1_tile                          # fills/drains + operand re-reads
    return {"macs": macs, "vec_flops": vec, "hbm_bytes": hbm, "l2_bytes": l2, "link_bytes": link}


def lm_head_counts(cfg, batch: int, dev: int, w: int = 2) -> Dict[str, float]:
    n = cfg.vocab // dev
    macs = batch * cfg.d_model * n
    hbm = (batch * cfg.d_model + cfg.d_model * n + batch * n) * w
    return {"macs": macs, "vec_flops": 0.0, "hbm_bytes": hbm, "l2_bytes": 2 * hbm, "link_bytes": 0.0}


def add_counts(*cs, scale=None):
    out = {k: 0.0 for k in cs[0]}
    for i, c in enumerate(cs):
        m = 1 if scale is None else scale[i]
        for k in out:
            out[k] += m * c[k]
    return out


def calibrate_e_mac(params: EnergyParams, gemm_macs: float, gemm_hbm_bytes: float,
                    gemm_latency_s: float, target_w: float = 400.0) -> EnergyParams:
    """Solve e_mac so a compute-bound GEMM at f0 draws `target_w` (one device)."""
    l2 = 2 * gemm_hbm_bytes + 2 * gemm_macs * 2 / params.l1_tile
    other = params.energy({"macs": 0, "vec_flops": 0, "hbm_bytes": gemm_hbm_bytes,
                           "l2_bytes": l2, "link_bytes": 0}, gemm_latency_s, params.f0)
    budget_j = target_w * gemm_latency_s - sum(other.values())
    return replace(params, e_mac_pj=budget_j / gemm_macs * 1e12)
