Reference: TensorRT-LLM v0.5.0, A100 FP16, Llama-2-70B, TP=4

| Metric | Published | Simulated | Error |
|---|---:|---:|---:|
| TTFT, batch 1, 128-token prompt (ms) | 47.0 | 52.7 | +12.0% |
| TTFT, batch 1, 2048-token prompt (ms) | 377.0 | 322.6 | -14.4% |
| Throughput, batch 64, 128 in / 128 out (out tok/s) | 1,237 | 1,081 | -12.6% |
| Throughput, batch 64, 128 in / 2048 out (out tok/s) | 1,181 | 1,179 | -0.2% |
| Throughput, batch 64, 2048 in / 128 out (out tok/s) | 272 | 313 | +15.0% |
| Throughput, batch 64, 2048 in / 2048 out (out tok/s) | 738 | 931 | +26.1% |

Mean absolute error: 13.4%

Batch-1 decode step at 2K context (no published A100 TPOT to compare): 48.2 ms
