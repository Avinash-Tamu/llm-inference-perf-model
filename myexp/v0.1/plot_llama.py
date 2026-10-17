"""Plots for the Llama-3-70B study. Run after sweep_llama.py:  python myexp/v0.1/plot_llama.py"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "myexp/v0.1/results"
F = "myexp/v0.1/figures"
os.makedirs(F, exist_ok=True)

# palette (validated reference categorical slots 1-3) + neutral ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# A100 per device (LLMCompass GA100 config)
PEAK_TFLOPS = 108 * 4 * 16 * 16 * 2 * 1.41e9 / 1e12  # 312 TFLOP/s systolic
HBM_TBS = 5 * 1024 * 3.2e9 / 8 / 1e12  # 2.048 TB/s: 5 active HBM2e channels x 1024 pins x 3.2 Gb/s

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})


def load(name):
    with open(os.path.join(R, name)) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                r[k] = {"True": True, "False": False}.get(v, v)
    return rows


dec, pre, ops = load("decode_sweep.csv"), load("prefill_sweep.csv"), load("op_stats.csv")

# ---------- Figure 1: roofline ----------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ai = [0.5, 2000]
ridge = PEAK_TFLOPS / HBM_TBS
ax.plot([ai[0], ridge, ai[1]], [ai[0] * HBM_TBS, PEAK_TFLOPS, PEAK_TFLOPS], color=INK2, lw=2)
ax.annotate(f"ridge ≈ {ridge:.0f} FLOP/B", (ridge, PEAK_TFLOPS), xytext=(ridge * 1.6, PEAK_TFLOPS * 0.25),
            color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

WEIGHT = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
groups = [  # drawn in this order; MHA shows only attention (its weight matmuls equal GQA's)
    ("Prefill, s=2048", lambda o: o["phase"] == "prefill" and o["seq"] == 2048, AQUA, ("weight", "attn")),
    ("Decode GQA, b=64, 4K ctx", lambda o: o["phase"] == "decode" and o["variant"] == "GQA" and o["batch"] == 64 and o["seq"] == 4096, BLUE, ("weight", "attn")),
    ("Decode MHA baseline, b=64, 4K ctx", lambda o: o["phase"] == "decode" and o["variant"] == "MHA" and o["batch"] == 64 and o["seq"] == 4096, ORANGE, ("attn",)),
]
for label, pred, col, kinds in groups:
    sel = [o for o in ops if pred(o)]
    for kind in kinds:
        marker = "o" if kind == "weight" else "D"
        pts = [o for o in sel if (o["op"] in WEIGHT) == (kind == "weight")]
        x = [o["flops"] / o["bytes"] for o in pts]
        y = [o["flops"] / (o["latency_us"] * 1e-6) / 1e12 for o in pts]
        ax.scatter(x, y, s=46, marker=marker, color=col, edgecolor=SURF, linewidth=1.5, zorder=3,
                   label=label if kind == kinds[0] else None)
ax.annotate("GQA attention: AI ≈ group size (8)", (7.5, 11.8), xytext=(1.2, 40), fontsize=8.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.annotate("MHA attention: AI ≈ 1", (1.0, 1.95), xytext=(1.5, 0.9), fontsize=8.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.scatter([], [], marker="o", color=INK2, label="weight matmuls (QKV/O/FFN)")
ax.scatter([], [], marker="D", color=INK2, label="attention matmuls (QK, AV)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(*ai); ax.set_ylim(0.5, 600)
ax.set_xlabel("Arithmetic intensity (FLOP / byte of DRAM traffic)")
ax.set_ylabel("Achieved TFLOP/s per A100")
ax.set_title("Llama-3-70B on 4×A100 (TP=4): prefill is compute-bound, decode is not", loc="left", fontsize=11, color=INK)
ax.legend(loc="upper left", fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{F}/roofline.png", dpi=160); plt.close(fig)

# ---------- Figure 2: decode throughput vs batch (small multiples by context) ----------
ctxs = sorted({r["ctx"] for r in dec})
fig, axes = plt.subplots(1, len(ctxs), figsize=(7.2, 3.6), sharey=True)
axes = axes if len(ctxs) > 1 else [axes]
for ax, ctx in zip(axes, ctxs):
    for var, col in (("GQA", BLUE), ("MHA", ORANGE)):
        rs = sorted([r for r in dec if r["variant"] == var and r["ctx"] == ctx], key=lambda r: r["batch"])
        xb = [r["batch"] for r in rs]; yt = [r["throughput_tok_s"] for r in rs]
        ax.plot(xb, yt, color=col, lw=2, zorder=2)
        fit = [r for r in rs if r["fits_in_hbm"]]; nofit = [r for r in rs if not r["fits_in_hbm"]]
        ax.scatter([r["batch"] for r in fit], [r["throughput_tok_s"] for r in fit], s=40, color=col,
                   edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.scatter([r["batch"] for r in nofit], [r["throughput_tok_s"] for r in nofit], s=40,
                   facecolor=SURF, edgecolor=col, linewidth=1.5, zorder=3)
        ax.annotate(var, (xb[-1], yt[-1]), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=9, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 32, 64, 128, 256]); ax.set_xticklabels(["1", "8", "32", "64", "128", "256"])
    ax.set_title(f"{int(ctx/1024)}K context", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("Batch size")
axes[0].set_ylabel("Decode throughput (tokens/s)")
fig.suptitle("Decode throughput: GQA vs. MHA baseline  (hollow = KV cache doesn't fit in 4×80 GB)",
             x=0.01, ha="left", fontsize=10.5, color=INK)
fig.tight_layout(); fig.savefig(f"{F}/decode_throughput.png", dpi=160); plt.close(fig)

# ---------- Figure 3: where decode time goes (GQA) ----------
rs = sorted([r for r in dec if r["variant"] == "GQA"], key=lambda r: (r["ctx"], r["batch"]))
labels = [f"b={int(r['batch'])}\n{int(r['ctx']/1024)}K" for r in rs]
w = [r["weight_matmul_share"] * 100 for r in rs]
a = [r["attn_share"] * 100 for r in rs]
o = [100 - x - y for x, y in zip(w, a)]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
xs = range(len(rs))
ax.bar(xs, w, color=BLUE, width=0.7, label="weight matmuls", edgecolor=SURF, linewidth=2)
ax.bar(xs, a, bottom=w, color=AQUA, width=0.7, label="attention (QK, softmax, AV)", edgecolor=SURF, linewidth=2)
ax.bar(xs, o, bottom=[x + y for x, y in zip(w, a)], color=GRID, width=0.7,
       label="norms, activations, all-reduce, launch overhead", edgecolor=SURF, linewidth=2)
ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("% of layer decode time"); ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
ax.set_title("GQA decode time breakdown per layer", loc="left", fontsize=11, color=INK)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
fig.tight_layout(); fig.savefig(f"{F}/decode_breakdown.png", dpi=160); plt.close(fig)
print("figures written to", F)
