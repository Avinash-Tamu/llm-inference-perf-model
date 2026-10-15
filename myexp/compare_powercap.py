"""
Check the v0.2 energy/DVFS model against published A100 power-capping measurements,
and test which modeling assumption explains the gap.

Reference: Samsi et al., "From Words to Watts: Benchmarking the Energy Costs of Large
Language Model Inference", IEEE HPEC 2023 (arXiv:2310.03003), Table III:
LLaMA-65B on 4x A100-80GB, batch 64, relative to a 250 W cap baseline.
  175 W cap: +6.7% time, -23.2% energy (average over output lengths 256/512/1024)
  150 W cap: +19.5% time, -34.0% energy
Implied uncapped power: at 175 W, power fell by (1-0.232)/(1+0.067) = 0.72x, so the
baseline drew ~175/0.72 = 243 W per GPU.

A power cap runs the GPU at the highest clock whose power fits under the cap. We
emulate that: find f where modeled power = cap (interpolated over the simulated clock
grid, clamped at 0.60 GHz) and compare with the baseline (highest clock <= 250 W).

Scenarios (all reuse the cached LLMCompass runs from dvfs_sweep.py). alpha = fraction of
LLMCompass's fixed per-kernel overhead (26.9 ms per decode step) treated as GPU-side work
that stretches with 1/f; the rest is CPU-side launch latency.
  A    as first built: alpha = 0, no HBM-side data-path energy.
  B    alpha = 1, no HBM-side data-path energy.
  C(a) alpha = a, HBM-side data-path energy e_mempath fitted so uncapped power = 243 W.
The fit uses only the implied baseline power; the four published deltas are not fitted.
Choosing alpha among {0, 0.5, 1} does use them, so it is reported as a sensitivity and
the selected alpha is labeled as such.

Run after myexp/dvfs_sweep.py. Writes myexp/results/powercap_check.md and powercap_fit.json
"""
import json
from energy_common import load_cache, params_for, curve

PUB = {175: (6.7, -23.2), 150: (19.5, -34.0)}
P_BASE_IMPLIED = 175 / ((1 - 0.232) / (1 + 0.067))
cache = load_cache()


def at_power(pts, cap):
    if cap >= pts[-1]["power_W_per_gpu"]:
        return pts[-1]
    if cap <= pts[0]["power_W_per_gpu"]:
        return pts[0]
    for a, b in zip(pts, pts[1:]):
        if a["power_W_per_gpu"] <= cap <= b["power_W_per_gpu"]:
            w = (cap - a["power_W_per_gpu"]) / (b["power_W_per_gpu"] - a["power_W_per_gpu"])
            return {k: a[k] * (1 - w) + b[k] * w for k in ("f_GHz", "latency_ms", "J_per_token", "power_W_per_gpu")}


def fit_mempath(target_w, alpha):
    lo, hi = 0.0, 1000.0
    for _ in range(50):
        mid = (lo + hi) / 2
        p = curve(cache, "wtw_65b", params_for(cache, e_mempath_pj_per_byte=mid), alpha)[-1]["power_W_per_gpu"]
        lo, hi = (mid, hi) if p < target_w else (lo, mid)
    return (lo + hi) / 2


def deltas(pts):
    base = at_power(pts, 250)
    res = {}
    for cap in PUB:
        c = at_power(pts, cap)
        res[cap] = (c["f_GHz"], 100 * (c["latency_ms"] / base["latency_ms"] - 1),
                    100 * (c["J_per_token"] / base["J_per_token"] - 1))
    return res


if __name__ == "__main__":
    SCEN = [("A", "as first built", 0.0, None), ("B", "alpha=1, no data-path energy", 1.0, None)]
    fits = {}
    for a in (0.0, 0.5, 1.0):
        fits[a] = fit_mempath(P_BASE_IMPLIED, a)
        SCEN.append((f"C({a:g})", f"alpha={a:g}, data path {fits[a]:.0f} pJ/B (fitted)", a, fits[a]))
    out = ["Reference: Samsi et al., HPEC 2023 (From Words to Watts), Table III: LLaMA-65B, 4x A100-80GB, batch 64.",
           f"Implied uncapped power (from the 175 W row): {P_BASE_IMPLIED:.0f} W/GPU.",
           "",
           "| Scenario | Power at 1.41 GHz | Cap | Clock | Time: measured / modeled | Energy: measured / modeled |",
           "|---|---:|---:|---:|---:|---:|"]
    err = {}
    for tag, desc, a, mp in SCEN:
        params = params_for(cache, e_mempath_pj_per_byte=mp or 0.0)
        pts = curve(cache, "wtw_65b", params, a)
        d = deltas(pts)
        err[tag] = sum(abs(d[c][1] - PUB[c][0]) + abs(d[c][2] - PUB[c][1]) for c in PUB) / 4
        for cap, (dt, de) in PUB.items():
            fc, mt, me = d[cap]
            out.append(f"| {tag}: {desc} | {pts[-1]['power_W_per_gpu']:.0f} W | {cap} W | {fc:.2f} GHz | "
                       f"{dt:+.1f}% / {mt:+.1f}% | {de:+.1f}% / {me:+.1f}% |")
    out.append("")
    out.append("Mean absolute error over the four deltas (percentage points): " +
               ", ".join(f"{k} {v:.1f}" for k, v in err.items()))
    best = min((k for k in err if k.startswith("C")), key=err.get)
    alpha = float(best[2:-1])
    out.append(f"Selected for the calibrated model: {best} (alpha chosen using the deltas, see note above).")
    open("myexp/results/powercap_check.md", "w").write("\n".join(out) + "\n")
    pc = params_for(cache, e_mempath_pj_per_byte=fits[alpha])
    json.dump({"alpha": alpha, "e_mempath_pj_per_byte": fits[alpha], "e_mac_pj": pc.e_mac_pj,
               "p_base_implied_w": P_BASE_IMPLIED, "mae_pp": err},
              open("myexp/results/powercap_fit.json", "w"), indent=1)
    print("\n".join(out))
