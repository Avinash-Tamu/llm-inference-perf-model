"""
Validate the Llama GQA model against published measurements.

Reference: NVIDIA TensorRT-LLM v0.5.0 performance tables, "A100 GPUs (FP16)", LLaMA 70B, TP=4
https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.5.0/docs/source/performance.md
That release benchmarked Llama-2-70B (GQA, 8 KV heads), so we simulate LLAMA2_70B
(identical layer shapes to Llama-3-70B, vocab 32000).

How each published metric is reproduced:
  * 1st-token latency (batch 1): 80 x prefill-layer latency + LM head.
  * Throughput (batch 64, out tok/s): TensorRT-LLM's session benchmark divides
    batch x output_len by the full request time, prefill included. So
      total = prefill(batch 64, input_len) + sum_{i<output_len} decode_step(64, input_len + i)
    decode_step is simulated at a few context lengths and interpolated linearly
    (its only context-dependent part, attention over the KV cache, is linear in context).
    Batch-64 prefill is simulated as one call when cheap; for 2048-token prompts it is
    simulated at batch 4 and scaled by 16 (compute-bound: matmul M is already 8192).

Run from repo root:  python myexp/validate_published.py
Writes myexp/results/validation.csv and myexp/results/validation.md
"""
import csv, json, os, sys, time
from multiprocessing import Pool

sys.path.insert(0, ".")
os.makedirs("systolic_array_model/temp", exist_ok=True)
os.makedirs("myexp/results", exist_ok=True)

from design_space_exploration.dse import template_to_system, read_architecture_template
from software_model.llama import LLAMA2_70B, LlamaDecoderLayerTP, LMHeadTP
from software_model.utils import Tensor

TP, CFG = 4, LLAMA2_70B
CACHE = "myexp/results/validation_points.json"
REF = "TensorRT-LLM v0.5.0, A100 FP16, Llama-2-70B, TP=4"

# Published numbers (verbatim from the reference tables)
PUB_TTFT = {128: 47.0, 2048: 377.0}  # ms, batch 1
PUB_TPUT = {(128, 128): 1237, (128, 2048): 1181, (2048, 128): 272, (2048, 2048): 738}  # out tok/s, batch 64
BATCH = 64

DECODE_CTX = [128, 256, 1152, 2176, 3136, 4096]
PREFILL = [(1, 128), (1, 2048), (64, 128), (4, 2048)]

_sys = None


def system():
    global _sys
    if _sys is None:
        _sys = template_to_system(read_architecture_template("configs/GA100.json"))
    return _sys


def head_latency(batch):
    h = LMHeadTP(CFG, TP)
    h(Tensor([batch, 1, CFG.d_model]))
    return h.compile_and_simulate(system())


def point(job):
    kind, b, n = job
    t0 = time.time()
    L = LlamaDecoderLayerTP(CFG, TP)
    if kind == "prefill":
        L(Tensor([b, n, CFG.d_model]), "prefill")
    else:
        L(Tensor([b, 1, CFG.d_model]), "decode", kv_len=n - 1)
    layer = L.compile_and_simulate(system())
    total = CFG.n_layers * layer + head_latency(b)
    print(f"[{kind}] b={b} n={n}: {total*1e3:.2f} ms ({time.time()-t0:.0f}s)", flush=True)
    return f"{kind},{b},{n}", total


def interp(xs, ys, x):
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            w = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] * (1 - w) + ys[i + 1] * w
    raise ValueError(x)


if __name__ == "__main__":
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    jobs = [("prefill", b, s) for b, s in PREFILL] + [("decode", BATCH, c) for c in DECODE_CTX] + [("decode", 1, 2048)]
    todo = [j for j in jobs if f"{j[0]},{j[1]},{j[2]}" not in cache]
    if todo:
        with Pool(2) as pool:
            for k, v in pool.imap_unordered(point, todo):
                cache[k] = v
                json.dump(cache, open(CACHE, "w"), indent=1)

    step = [cache[f"decode,{BATCH},{c}"] for c in DECODE_CTX]
    prefill64 = {128: cache["prefill,64,128"], 2048: 16 * cache["prefill,4,2048"]}

    rows = []
    for s, pub in PUB_TTFT.items():
        sim = cache[f"prefill,1,{s}"] * 1e3
        rows.append({"metric": f"TTFT, batch 1, {s}-token prompt", "unit": "ms",
                     "published": pub, "simulated": round(sim, 1), "error_pct": round(100 * (sim - pub) / pub, 1)})
    for (i, o), pub in PUB_TPUT.items():
        dec = sum(interp(DECODE_CTX, step, i + k) for k in range(o))
        total = prefill64[i] + dec
        sim = BATCH * o / total
        rows.append({"metric": f"Throughput, batch 64, {i} in / {o} out", "unit": "out tok/s",
                     "published": pub, "simulated": round(sim), "error_pct": round(100 * (sim - pub) / pub, 1),
                     "prefill_share_pct": round(100 * prefill64[i] / total, 1)})

    with open("myexp/results/validation.csv", "w", newline="") as f:
        keys = ["metric", "unit", "published", "simulated", "error_pct", "prefill_share_pct"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    mape = sum(abs(r["error_pct"]) for r in rows) / len(rows)
    with open("myexp/results/validation.md", "w") as f:
        f.write(f"Reference: {REF}\n\n| Metric | Published | Simulated | Error |\n|---|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['metric']} ({r['unit']}) | {r['published']:,} | {r['simulated']:,} | {r['error_pct']:+.1f}% |\n")
        f.write(f"\nMean absolute error: {mape:.1f}%\n")
        f.write(f"\nBatch-1 decode step at 2K context (no published A100 TPOT to compare): "
                f"{cache['decode,1,2048']*1e3:.1f} ms\n")
    print(open("myexp/results/validation.md").read())
