# Llama-3-70B inference on systolic-array accelerators: GQA, KV cache and the memory wall

An extension of [LLMCompass](https://github.com/PrincetonUniversity/LLMCompass) (Zhang et al., ISCA 2024),
a validated hardware simulator for LLM inference. Upstream LLMCompass models GPT-3 (multi-head attention,
GeLU FFN). This fork adds a **Llama-3-style decoder layer**: grouped-query attention (GQA), SwiGLU FFN
and RMSNorm. It uses the fork to answer one question:

> **On a 4×A100 node serving Llama-3-70B, what actually limits decode throughput, and how much of the
> answer is GQA?**

Everything below comes from LLMCompass's tile-level simulator (systolic-array cycle counts from SCALE-Sim,
L2/DRAM tiling, NVLink all-reduce). It uses public model configs and public hardware specs only.

## Key findings

Setup: Llama-3-70B, FP16, tensor parallel = 4 on 4× A100-80GB (LLMCompass `GA100` config:
312 TFLOP/s and 2.05 TB/s HBM per GPU, so the ridge point is 152 FLOP/byte).

1. **Decode is memory-bound, not compute-bound.** At 4K context, going from batch 1 to batch 64 raises
   per-token latency by only 29% (48.5 → 62.8 ms) while throughput rises 48× (21 → 1,019 tokens/s).
   Weight matmuls run at arithmetic intensity ≈ batch size, far left of the 152 FLOP/B ridge.
   Prefill sits at or near the compute roof (TTFT: 100 ms for 512 tokens, 323 ms for 2K, 1.57 s for 8K).
2. **Attention is memory-bound at every batch size.** Batching does not help it: attention's arithmetic
   intensity equals the GQA group size (8, or 1 for MHA), whatever the batch. At 32K context, attention
   is 28% of layer time at batch 8 and 54% at batch 32 (past the memory limit), overtaking the weight matmuls.
3. **GQA's biggest win is capacity, not bandwidth.** At the same batch, GQA is about 2.2× faster than an
   MHA version of the same model (4K, b=64: 1,019 vs 462 tokens/s). But the KV cache shrinks 8×
   (320 KB vs 2.5 MB per token), so after the 35 GB/GPU of weights, GQA fits batch ~121 at 4K (MHA: ~15)
   and batch ~15 at 32K (MHA: 1). **Comparing the largest batch that fits, GQA serves about 8× more
   tokens/s** (4K: 1,019 vs 130; 32K: 131 vs 16).
4. **Long context turns into a KV-cache-bandwidth problem.** At 32K context, GQA throughput flattens
   around ~500 tokens/s even with unlimited memory. Past that point, faster HBM or smaller KV (FP8 KV,
   MLA, sliding window) matters more than more FLOPs.
5. **Fixed per-kernel overheads matter at small batch.** At batch 1, norms, activations, all-reduce and
   kernel-launch overhead are about 30% of layer time. Largely because of them, simulated decode latency at 4K
   context is 20–30% above the pure roofline.

![Roofline](myexp/figures/roofline.png)
![Decode throughput](myexp/figures/decode_throughput.png)
![Decode breakdown](myexp/figures/decode_breakdown.png)

Raw numbers: [`myexp/results/`](myexp/results/) (`decode_sweep.csv`, `prefill_sweep.csv`, `op_stats.csv`).

## What I added

| File | What it is |
|---|---|
| [`software_model/llama.py`](software_model/llama.py) | `LlamaDecoderLayerTP`: GQA attention modeled as a batched matmul `[b·n_kv, g·s, d_h] × [b·n_kv, d_h, s_kv]`, so the query group becomes the matmul M dimension; SwiGLU FFN; RMSNorm; TP all-reduces; per-op FLOP/byte accounting. `LMHeadTP` for the vocab projection. `LlamaConfig` with public Llama-3-70B parameters and an MHA counterfactual. |
| [`myexp/validate_llama.py`](myexp/validate_llama.py) | Hand-calculation checks: 70.55 B parameters, 320 KB KV cache per token (2·80·8·128·2 B), 34.2 GB weights per GPU at TP=4, attention arithmetic intensity ≈ 8. |
| [`myexp/sweep_llama.py`](myexp/sweep_llama.py) | Decode sweep (batch 1–256 × context 4K/32K × GQA/MHA) and prefill sweep (512–8K), with HBM capacity check. |
| [`myexp/plot_llama.py`](myexp/plot_llama.py) | The three figures above. |

## Reproduce

```bash
conda create -n llmcompass python=3.11 -y && conda activate llmcompass
pip install torch "scalesim==2.0.2" scipy matplotlib seaborn pandas numpy
python myexp/validate_llama.py      # seconds
python myexp/sweep_llama.py         # ~1 h on 2 cores the first run (cached afterwards)
python myexp/plot_llama.py
```
`scalesim` must be 2.0.x. Version 3.0 changed its config format and breaks LLMCompass.

## Modeling assumptions and limitations

- **Memory-capacity check:** 80 GB/GPU − FP16 weights/TP − 4 GB reserve for activations and workspace.
  Throughput beyond that line is reported (hollow markers) to show the bandwidth trend, but a real
  server can't run it without offloading or KV compression.
- No causal-mask savings in prefill. No FlashAttention-style fusion: softmax scores round-trip through
  memory, so long-context attention is somewhat pessimistic.
- SiLU uses LLMCompass's GeLU vector-op cost, and RMSNorm uses LayerNorm's (slightly pessimistic).
  Reshape, transpose and KV-append are not simulated, consistent with upstream.
- No speculative decoding, paged-KV fragmentation or continuous-batching scheduler effects.
- LLMCompass's own validation (upstream paper) reports ~4% average error on end-to-end LLM inference
  against real A100s. The Llama-specific numbers here have not been validated against hardware.

## Roadmap

- **v0.2: energy model.** Per-operator dynamic energy (MACs, SRAM, HBM) plus leakage, and DVFS
  operating points. Decode is memory-bound, so core V/f can drop with little latency cost. The goal
  is to quantify energy per token against latency.
- Disaggregated prefill/decode hardware design-space exploration.

## Credit

Built on LLMCompass by Hengrui Zhang, August Ning, Rohan Baskar Prabhakar and David Wentzlaff
(Princeton): [paper](https://parallel.princeton.edu/papers/isca24_llmcompass.pdf). The original README is in
[`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md). License: BSD 3-Clause, same as upstream (see [LICENSE](LICENSE)).
