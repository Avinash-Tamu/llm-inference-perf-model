# v0.2: Energy per token under core DVFS

**Question:** decode is memory-bound, so how much energy per token does lowering the core clock save,
and at what latency cost? How does prefill compare?

The energy model itself lives in [`hardware_model/energy_model.py`](../../hardware_model/energy_model.py).

| Step | Script | What it does | Output |
|---|---|---|---|
| 1 | `dvfs_sweep.py` | Reruns LLMCompass at 7 core clocks (0.6–1.41 GHz) for 4 workloads, plus one GEMM for calibration. Slow (~2–3 h), cached per point. | `results/dvfs_points.json` |
| 2 | `compare_powercap.py` | Checks the model against published A100 power-capping data (Samsi et al., HPEC 2023) and fits the HBM data-path energy | `results/powercap_check.md`, `powercap_fit.json` |
| 3 | `energy_report.py` | Latency, W/GPU, J/token and energy breakdown per workload and clock, for the calibrated and as-built models | `results/dvfs_energy.csv` |
| 4 | `plot_energy.py` | Energy vs latency (DVFS Pareto) and energy breakdown | `figures/*.png` |
| – | `energy_common.py` | Shared helpers (workloads, clock grid, energy curve) | – |

**Results in one line each**
- Model as first built failed the measured check; after one fitted parameter it matches within 3.1 pp.
- Decode: 1.41 → 1.2 GHz costs +4.4% latency for −14% energy per token (about 3:1).
- Prefill: the same clock drop is about 1:1 (+12% / −14%). Run prefill fast and decode slow.

Run from the repo root, e.g. `python myexp/v0.2/compare_powercap.py`. Full write-up: [main README](../../README.md).
