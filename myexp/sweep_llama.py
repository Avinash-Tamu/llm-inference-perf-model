"""
Sweeps for Llama-3-70B on 4x A100-80GB (tensor parallel = 4), GQA vs. MHA baseline.

Outputs (myexp/results/):
  decode_sweep.csv   simulated + roofline per-token decode latency over batch x context
  prefill_sweep.csv  simulated prefill latency over prompt length
  op_stats.csv       per-operator FLOPs / bytes / latency for roofline plots

Run from repo root:  python myexp/sweep_llama.py [--quick]
Full run takes ~1 hour on 2 cores the first time (scalesim results are cached in
systolic_array_model/look_up_table_16_16.csv, so reruns are much faster).
"""
import csv, os, sys, time
from multiprocessing import Pool

sys.path.insert(0, ".")
os.makedirs("systolic_array_model/temp", exist_ok=True)
os.makedirs("myexp/results", exist_ok=True)

from design_space_exploration.dse import template_to_system, read_architecture_template
from software_model.llama import LLAMA3_70B, LLAMA3_70B_MHA, LlamaDecoderLayerTP, LMHeadTP
from software_model.utils import Tensor

TP = 4
HBM_PER_GPU = 80e9
RESERVE = 4e9  # activations, workspace, fragmentation
CONFIGS = {"GQA": LLAMA3_70B, "MHA": LLAMA3_70B_MHA}
MATMUL_OPS = LlamaDecoderLayerTP.MATMULS
ATTN_OPS = ["qk", "softmax", "av"]
WEIGHT_OPS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

_system = None


def system():
    global _system
    if _system is None:
        _system = template_to_system(read_architecture_template("configs/GA100.json"))
    return _system


def kv_fits(cfg, batch, ctx):
    weights = cfg.param_count() * 2 / TP
    kv = batch * ctx * cfg.kv_bytes_per_token() / TP
    return weights + kv + RESERVE <= HBM_PER_GPU, kv / 1e9


def decode_point(args):
    variant, batch, ctx = args
    cfg, sysm = CONFIGS[variant], system()
    t0 = time.time()
    L = LlamaDecoderLayerTP(cfg, TP)
    L(Tensor([batch, 1, cfg.d_model]), "decode", kv_len=ctx - 1)
    roof = L.roofline_model(sysm)
    sim = L.compile_and_simulate(sysm)
    bd = dict(L.breakdown)
    head = LMHeadTP(cfg, TP)
    head(Tensor([batch, 1, cfg.d_model]))
    head_t = head.compile_and_simulate(sysm)
    tok = cfg.n_layers * sim + head_t
    fits, kv_gb = kv_fits(cfg, batch, ctx)
    stats = L.op_stats()
    row = {
        "variant": variant, "batch": batch, "ctx": ctx,
        "layer_sim_us": sim * 1e6, "layer_roofline_us": roof * 1e6,
        "token_latency_ms": tok * 1e3,
        "throughput_tok_s": batch / tok,
        "attn_share": sum(bd[k] for k in ATTN_OPS) / sim,
        "weight_matmul_share": sum(bd[k] for k in WEIGHT_OPS) / sim,
        "kv_gb_per_gpu": kv_gb, "fits_in_hbm": fits,
        "wall_s": time.time() - t0,
    }
    ops = []
    for name in MATMUL_OPS:
        ops.append({"phase": "decode", "variant": variant, "batch": batch, "seq": ctx, "op": name,
                    "flops": stats[name]["flops"], "bytes": stats[name]["bytes"],
                    "latency_us": bd[name] * 1e6})
    print(f"[decode] {variant} b={batch} ctx={ctx}: {tok*1e3:.1f} ms/token, "
          f"{batch/tok:.0f} tok/s, fits={fits} ({time.time()-t0:.0f}s)", flush=True)
    return row, ops


def prefill_point(args):
    variant, seq = args
    cfg, sysm = CONFIGS[variant], system()
    t0 = time.time()
    L = LlamaDecoderLayerTP(cfg, TP)
    L(Tensor([1, seq, cfg.d_model]), "prefill")
    roof = L.roofline_model(sysm)
    sim = L.compile_and_simulate(sysm)
    bd = dict(L.breakdown)
    stats = L.op_stats()
    row = {"variant": variant, "seq": seq, "layer_sim_us": sim * 1e6, "layer_roofline_us": roof * 1e6,
           "ttft_ms": cfg.n_layers * sim * 1e3,
           "attn_share": sum(bd[k] for k in ATTN_OPS) / sim, "wall_s": time.time() - t0}
    ops = [{"phase": "prefill", "variant": variant, "batch": 1, "seq": seq, "op": n,
            "flops": stats[n]["flops"], "bytes": stats[n]["bytes"], "latency_us": bd[n] * 1e6}
           for n in MATMUL_OPS]
    print(f"[prefill] {variant} s={seq}: TTFT {row['ttft_ms']:.0f} ms ({time.time()-t0:.0f}s)", flush=True)
    return row, ops


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    batches = [1, 16, 64] if quick else [1, 8, 32, 64, 128, 256]
    ctxs = [4096] if quick else [4096, 32768]
    seqs = [512] if quick else [512, 2048, 8192]
    decode_jobs = [(v, b, c) for v in CONFIGS for c in ctxs for b in batches]
    prefill_jobs = [("GQA", s) for s in seqs]
    with Pool(2) as pool:
        pre = pool.map(prefill_point, prefill_jobs, chunksize=1)
        dec = pool.map(decode_point, decode_jobs, chunksize=1)
    write("myexp/results/decode_sweep.csv", [r for r, _ in dec])
    write("myexp/results/prefill_sweep.csv", [r for r, _ in pre])
    write("myexp/results/op_stats.csv", [o for _, ops in pre + dec for o in ops])
    print("done")
