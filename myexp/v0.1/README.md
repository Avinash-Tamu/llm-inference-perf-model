# v0.1: Llama-3-70B with GQA on 4× A100

**Question:** what limits Llama-3-70B decode throughput on a 4×A100 node, and how much does GQA help?

| Script | What it does | Output |
|---|---|---|
| `validate_llama.py` | Hand-calculation checks (70.55 B params, 320 KB KV cache per token, per-op arithmetic intensity) | prints |
| `sweep_llama.py` | Decode sweep (batch 1–256 × context 4K/32K × GQA/MHA) and prefill sweep (512–8K) | `results/decode_sweep.csv`, `prefill_sweep.csv`, `op_stats.csv` |
| `plot_llama.py` | Roofline, decode throughput, decode time breakdown | `figures/*.png` |
| `validate_published.py` | Compares the simulator with NVIDIA's published TensorRT-LLM A100 numbers (Llama-2-70B, TP=4) | `results/validation.md`, `validation.csv` |

**Results in one line each**
- Decode is memory-bound: batch 1 → 64 gives 48× throughput for +29% per-token latency.
- GQA's main win is KV-cache capacity: about 8× more servable throughput than MHA at the memory limit.
- Validation: 13.4% mean absolute error over 6 published points. The worst miss is +26% at long context.

Run from the repo root, e.g. `python myexp/v0.1/sweep_llama.py`. Full write-up: [main README](../../README.md).
