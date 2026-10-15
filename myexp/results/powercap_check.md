Reference: Samsi et al., HPEC 2023 (From Words to Watts), Table III: LLaMA-65B, 4x A100-80GB, batch 64.
Implied uncapped power (from the 175 W row): 243 W/GPU.

| Scenario | Power at 1.41 GHz | Cap | Clock | Time: measured / modeled | Energy: measured / modeled |
|---|---:|---:|---:|---:|---:|
| A: as first built | 172 W | 175 W | 1.41 GHz | +6.7% / +0.0% | -23.2% / +0.0% |
| A: as first built | 172 W | 150 W | 1.23 GHz | +19.5% / +0.2% | -34.0% / -12.6% |
| B: alpha=1, no data-path energy | 172 W | 175 W | 1.41 GHz | +6.7% / +0.0% | -23.2% / +0.0% |
| B: alpha=1, no data-path energy | 172 W | 150 W | 1.26 GHz | +19.5% / +5.9% | -34.0% / -7.8% |
| C(0): alpha=0, data path 82 pJ/B (fitted) | 243 W | 175 W | 1.01 GHz | +6.7% / +0.5% | -23.2% / -27.6% |
| C(0): alpha=0, data path 82 pJ/B (fitted) | 243 W | 150 W | 0.83 GHz | +19.5% / +1.2% | -34.0% / -37.6% |
| C(0.5): alpha=0.5, data path 82 pJ/B (fitted) | 243 W | 175 W | 1.06 GHz | +6.7% / +8.2% | -23.2% / -22.2% |
| C(0.5): alpha=0.5, data path 82 pJ/B (fitted) | 243 W | 150 W | 0.91 GHz | +19.5% / +13.7% | -34.0% / -29.9% |
| C(1): alpha=1, data path 82 pJ/B (fitted) | 243 W | 175 W | 1.09 GHz | +6.7% / +14.1% | -23.2% / -18.0% |
| C(1): alpha=1, data path 82 pJ/B (fitted) | 243 W | 150 W | 0.96 GHz | +19.5% / +23.2% | -34.0% / -24.3% |

Mean absolute error over the four deltas (percentage points): A 17.6, B 17.4, C(0) 8.1, C(0.5) 3.1, C(1) 6.5
Selected for the calibrated model: C(0.5) (alpha chosen using the deltas, see note above).
