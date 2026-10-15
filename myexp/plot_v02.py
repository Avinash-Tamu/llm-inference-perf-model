"""Figures for v0.2 (energy + DVFS), calibrated model. Run after myexp/energy_report.py."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [r for r in csv.DictReader(open("myexp/results/dvfs_energy.csv")) if r["model"] == "calibrated"]
for r in rows:
    for k, v in r.items():
        if k not in ("workload", "model"):
            r[k] = float(v)

SERIES = [("decode_b1", "Decode, batch 1", "#2a78d6"),
          ("decode_b64", "Decode, batch 64", "#eb6834"),
          ("prefill_2k", "Prefill, 2K prompt", "#1baf7a")]
COMP = [("hbm", "HBM device", "#2a78d6"), ("mempath", "HBM data path (on-chip)", "#eb6834"),
        ("static", "Static + clock", "#1baf7a"), ("compute", "Compute", "#eda100"),
        ("sram", "Operand SRAM", "#e87ba4")]  # NVLink < 1% everywhere: omitted
INK, MUTED = "#2b2b2b", "#77766f"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})

# 1) energy/token vs latency, normalized to 1.41 GHz
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for key, label, color in SERIES:
    pts = sorted([r for r in rows if r["workload"] == key], key=lambda r: r["f_GHz"])
    base = pts[-1]
    x = [p["latency_ms"] / base["latency_ms"] for p in pts]
    y = [p["J_per_token"] / base["J_per_token"] for p in pts]
    ax.plot(x, y, color=color, lw=2, marker="o", ms=6, mec="white", mew=1.5, label=label)
    ax.annotate(f"{pts[0]['f_GHz']:.2f} GHz", (x[0], y[0]), textcoords="offset points",
                xytext=(6, -2), fontsize=8, color=INK)
ax.axhline(1, color=MUTED, lw=0.6, ls=":")
ax.axvline(1, color=MUTED, lw=0.6, ls=":")
ax.annotate("1.41 GHz (nominal)", (1, 1), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK)
ax.set_xlabel("Latency relative to 1.41 GHz")
ax.set_ylabel("Energy per token relative to 1.41 GHz")
ax.set_title("Core DVFS on 4x A100: Llama-3-70B", color=INK, loc="left")
ax.grid(color="#e6e5df", lw=0.6)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("myexp/figures/dvfs_pareto.png", dpi=160)

# 2) energy breakdown at nominal clock
fig, ax = plt.subplots(figsize=(6.8, 3.4))
labels = []
for i, (key, label, _) in enumerate(SERIES):
    r = max((r for r in rows if r["workload"] == key), key=lambda r: r["f_GHz"])
    left = 0
    for c, cl, color in COMP:
        v = r[f"share_{c}"]
        ax.barh(i, v, left=left, color=color, edgecolor="white", linewidth=2, height=0.6,
                label=cl if i == 0 else None)
        if v > 0.07:
            ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white")
        left += v
    labels.append(f"{label}\n{r['J_per_token']:.3g} J/token")
ax.set_yticks(range(len(SERIES)), labels)
ax.invert_yaxis()
ax.set_xlim(0, 1.0)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_title("Where the energy goes at 1.41 GHz (4x A100, all GPUs)", color=INK, loc="left")
ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.16))
fig.tight_layout()
fig.savefig("myexp/figures/energy_breakdown.png", dpi=160)
print("saved")
