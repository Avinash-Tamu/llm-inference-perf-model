# LLMCompass: setup + code map for the energy/GQA extension

## Setup (Mac, Apple Silicon)
```
cd ~/Documents/BOT/LLMCompass
conda create -n llmcompass python=3.11 -y && conda activate llmcompass
pip install torch "scalesim==2.0.2" scipy matplotlib seaborn pandas numpy   # scalesim 3.x breaks LLMCompass
python myexp/smoke_test.py
```
Verified result (GPT-3 layer, 1x A100, ctx 2048, decode):
- bs=1  -> roofline 1.23 ms, simulated 2.14 ms
- bs=64 -> roofline 2.84 ms, simulated 5.35 ms
64x the batch costs only 2.5x the time, so decode is memory-bound. That is your first
data point for the roofline story.

Git: this is a full clone (7 commits + the `cost_model/supply_chain` submodule).
To make it your fork: create an empty repo on GitHub, then
`git remote rename origin upstream && git remote add origin <your-fork-url> && git push -u origin main`.

## Code map (where latency comes from)
| Layer | File | What matters |
|---|---|---|
| HW config (JSON) | `configs/GA100.json` | freq, cores, systolic array WxH, SRAM, global buffer, HBM channels x pins x Gb/s |
| JSON -> objects | `design_space_exploration/dse.py` `template_to_system()` | builds ComputeModule / IOModule / Device |
| Compute HW | `hardware_model/compute_module.py` | `ComputeModule.clock_freq` = the single clock for the whole chip; `l2_bandwidth_per_cycle` |
| Memory BW | `hardware_model/io_module.py` | `IOModule.bandwidth` (bytes/s), independent of core clock |
| Model graph | `software_model/transformer.py` | `TransformerBlockInitComputationTP` (prefill), `TransformerBlockAutoRegressionTP` (decode, with KV cache) |
| Matmul sim | `software_model/matmul.py` | `compile_and_simulate()` -> tiling mapper -> `L2TileSimulator` (DRAM<->L2 IO, line ~1009) and `L1TileSimulator` (systolic array, line ~1293); latency = cycles / clock_freq (line ~737) |
| Other ops | `softmax.py`, `layernorm.py`, `gelu.py`, `communication_primitives.py` | vector unit + all-reduce |

## Weekend 2 hooks: Llama-3-70B + GQA
Today the model is GPT-style: MHA, FFN = 4d with GeLU, LayerNorm. Llama-3 needs:
1. **GQA**: add `n_kv_heads` arg. `Wk`, `Wv` become `[d, n_kv_heads*d_h/dev]`; the K/V cache is
   `[b, n_kv_heads/dev, d_h, s]`. For Q_mul_K / A_mul_V, reshape Q to
   `[b, n_kv, (h/n_kv)*1, d_h]` so each KV head serves its query group. That is exactly why GQA
   decode has higher arithmetic intensity: the group dimension becomes the matmul M.
2. **SwiGLU FFN**: W1 (gate) and W3 (up) `[d, 28672/dev]`, W2 (down) `[28672/dev, d]`; reuse
   `GeLU` as a stand-in for SiLU (same vector-op cost class) plus an elementwise multiply.
3. **RMSNorm**: reuse `LayerNorm` (slightly pessimistic, fine for a first cut).
4. Copy both classes into a new `software_model/llama.py`; don't edit upstream classes, so
   you can still reproduce the paper.
Hand check: KV cache/token = 2 x 80 x 8 x 128 x 2 B = 320 KB (MHA: 2.5 MB).

## Weekend 4-5 hooks: energy + DVFS
- Every operator ends up in `Matmul.simulate()` / `L2TileSimulator` / `L1TileSimulator`,
  which already know the tile loop counts. Add counters there:
  `mac_count`, `sram_bytes` (L1 reads/writes), `l2_bytes`, `dram_bytes`.
- New file `hardware_model/energy_model.py`: pJ/MAC, pJ/bit per level, leakage W; scale
  dynamic energy by (V/V0)^2 and leakage by a V-dependent factor.
- DVFS: `clock_freq` scales compute cycles, but `IOModule.bandwidth` does not. So in
  memory-bound decode, lowering clock_freq barely changes latency. That is your headline
  result. Add a `set_operating_point(V, f)` that updates `clock_freq`
  (and `l2_bandwidth_per_cycle * f`) and returns the energy scale factors.
- Energy per op = dynamic counts x pJ + P_leak x latency. Sum per layer, then per token.
