| Config | Seeds | Test acc (seed sd) | Macro F1 | ECE | Real near-boundary |
|---|---|---|---|---|---|
| C_balanced | 1 | 94.3% ± 0.0% | 0.943 | 0.042 | 8/8 |
| D_balanced | 1 | 95.7% ± 0.0% | 0.959 | 0.054 | 8/8 |
| D_none | 3 | 96.9% ± 0.3% | 0.970 | 0.047 | 8/8, 8/8, 8/8 |

| Config | Sweep point | Strict | Tight ±6.25% | Tolerance step | One flip | Non-monotone | Max neg. step | First-crossing error (seed range) | Coverage |
|---|---|---|---|---|---|---|---|---|---|
| C_balanced | 0.976 | 0.958 | 0.902 | 100% | 100% | 39% | 0.279 | 0.0% [0.0, 0.0] | 100% |
| D_balanced | 0.937 | 0.930 | 0.742 | 100% | 100% | 0% | 0.001 | 0.9% [0.9, 0.9] | 100% |
| D_none | 0.947 | 0.947 | 0.783 | 100% | 100% | 0% | 0.002 | 0.6% [0.2, 1.1] | 100% |

Notes:
- `±` is the training-seed standard deviation on one fixed test set, not test-set uncertainty.
- First-crossing error is the first upward 0.5 crossing, interpolated in log-amount, over every case that has a crossing; `Coverage` is the fraction of cases with a crossing. It is not conditioned on the curve having exactly one flip. The bracketed range is the min/max of the per-seed medians, not a confidence interval; for a noisy curve (config A) the first crossing is an early noise crossing.
- `Tolerance step` requires the correct account at every amount outside a ±2% band around the boundary. The old `perfect step` also required the ±0.001% points and is 0% by construction.
- `Non-monotone` means any negative step in p(above) larger than 0.05; `Max neg. step` reports the largest such drop.
- `Real near-boundary` counts actual test items within ±12.5% of a rule boundary (not synthetic sweep points); it is shown per seed.
