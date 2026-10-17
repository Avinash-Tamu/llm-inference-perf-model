"""
v0.2 results: energy per token vs. core clock for Llama-3-70B on 4x A100 (TP=4),
under the calibrated model (scenario C of compare_powercap.py) and the as-built model (A).

Run after dvfs_sweep.py and compare_powercap.py. Writes myexp/v0.2/results/dvfs_energy.csv
"""
import csv, json
from energy_common import load_cache, params_for, curve, WORKLOADS

cache = load_cache()
fit = json.load(open("myexp/v0.2/results/powercap_fit.json"))
models = {"calibrated": (params_for(cache, e_mempath_pj_per_byte=fit["e_mempath_pj_per_byte"]), fit["alpha"]),
          "as_built": (params_for(cache), 0.0)}
rows = []
for mname, (params, scaled) in models.items():
    for w in WORKLOADS:
        for r in curve(cache, w, params, scaled):
            rows.append({"model": mname, **r})
with open("myexp/v0.2/results/dvfs_energy.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)

for r in rows:
    if r["model"] == "calibrated":
        print(f"{r['workload']:11s} f={r['f_GHz']:.3f} lat={r['latency_ms']:8.2f} ms "
              f"P={r['power_W_per_gpu']:6.1f} W/GPU E={r['J_per_token']:.4f} J/tok "
              f"hbm={r['share_hbm']:.2f} path={r['share_mempath']:.2f} sram={r['share_sram']:.2f} static={r['share_static']:.2f} "
              f"compute={r['share_compute']:.2f}")
