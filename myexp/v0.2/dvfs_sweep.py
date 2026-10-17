"""
v0.2 simulation runs: Llama layers on 4x A100-80GB (TP=4) at several core clocks.

LLMCompass is re-run with the device clock set to f, so compute cycles stretch while
HBM transfer time does not. Results are cached per point in myexp/v0.2/results/dvfs_points.json;
energy is computed afterwards from these latencies (see energy_report.py).

Workloads (defined in energy_common.py)
  decode_b1    Llama-3-70B decode, batch 1,  2K context   (latency-critical serving)
  decode_b64   Llama-3-70B decode, batch 64, 2K context   (throughput serving)
  prefill_2k   Llama-3-70B prefill, batch 1, 2K prompt    (compute-bound; coarser clock grid)
  wtw_65b      LLaMA-65B decode, batch 64, ~384 context   (compared with published A100
               power-capping measurements in compare_powercap.py)
Also simulates one 8192^3 FP16 GEMM at 1.41 GHz, used to calibrate MAC energy.

Run from repo root:  python myexp/v0.2/dvfs_sweep.py      (~2-3 h on 2 cores, first run)
Then:                python myexp/v0.2/compare_powercap.py && python myexp/v0.2/energy_report.py
"""
import copy, json, os, sys, time
from multiprocessing import Pool

sys.path.insert(0, ".")
os.makedirs("systolic_array_model/temp", exist_ok=True)
os.makedirs("myexp/v0.2/results", exist_ok=True)

from design_space_exploration.dse import template_to_system, read_architecture_template
from software_model.llama import LlamaDecoderLayerTP, LMHeadTP
from software_model.matmul import Matmul
from software_model.utils import Tensor, data_type_dict
from energy_common import TP, FREQS, FREQS_FOR, WORKLOADS, CACHE

_tmpl = read_architecture_template("configs/GA100.json")


def system_at(f):
    t = copy.deepcopy(_tmpl)
    t["device"]["frequency_Hz"] = f
    return template_to_system(t)


def sim_point(job):
    name, f = job
    cfg, mode, b, n = WORKLOADS[name]
    sysm, t0 = system_at(f), time.time()
    L = LlamaDecoderLayerTP(cfg, TP)
    if mode == "decode":
        L(Tensor([b, 1, cfg.d_model]), "decode", kv_len=n - 1)
    else:
        L(Tensor([b, n, cfg.d_model]), "prefill")
    layer_t = L.compile_and_simulate(sysm)
    head = LMHeadTP(cfg, TP)
    head(Tensor([b, 1, cfg.d_model]))
    head_t = head.compile_and_simulate(sysm)
    print(f"[{name}] f={f/1e9:.3f} GHz: layer {layer_t*1e6:.0f} us ({time.time()-t0:.0f}s)", flush=True)
    return f"{name},{f:.0f}", {"layer_s": layer_t, "head_s": head_t}


def gemm_point():
    mm = Matmul(data_type_dict["fp16"])
    n = 8192
    mm(Tensor([n, n]), Tensor([n, n]))
    t = mm.compile_and_simulate(system_at(1.41e9).device, "heuristic-GPU")
    print(f"[gemm] 8192^3 @1.41 GHz: {t*1e3:.2f} ms ({2*n**3/t/1e12:.0f} TFLOP/s)", flush=True)
    return {"latency_s": t, "n": n}


if __name__ == "__main__":
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if "gemm" not in cache:
        cache["gemm"] = gemm_point()
    jobs = [(w, f) for w in WORKLOADS for f in FREQS_FOR.get(w, FREQS) if f"{w},{f:.0f}" not in cache]
    with Pool(2) as pool:
        for k, v in pool.imap_unordered(sim_point, jobs):
            cache[k] = v
            json.dump(cache, open(CACHE, "w"), indent=1)
    json.dump(cache, open(CACHE, "w"), indent=1)
    print("done")
