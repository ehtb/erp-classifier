| Config | Seeds | Test acc (seed sd) | Macro F1 | ECE | Real near-boundary |
|---|---|---|---|---|---|
| C_balanced | 1 | 96.5% ± 0.0% | 0.962 | 0.052 | 8/8 |
| D_balanced | 1 | 98.6% ± 0.0% | 0.985 | 0.044 | 8/8 |
| D_none | 3 | 97.9% ± 1.5% | 0.978 | 0.039 | 8/8, 8/8, 8/8 |

| Config | Sweep point | Strict | Tight ±6.25% | Tolerance step | One flip | Non-monotone | Max neg. step | First-crossing error (seed range) | Coverage |
|---|---|---|---|---|---|---|---|---|---|
| C_balanced | 0.969 | 0.962 | 0.882 | 82% | 82% | 100% | 0.578 | 0.1% [0.1, 0.1] | 100% |
| D_balanced | 0.948 | 0.948 | 0.786 | 100% | 100% | 0% | 0.000 | 0.7% [0.7, 0.7] | 100% |
| D_none | 0.954 | 0.954 | 0.810 | 100% | 100% | 0% | 0.001 | 0.6% [0.4, 0.8] | 100% |

Notes:
- `±` is the training-seed standard deviation on one fixed test set, not test-set uncertainty.
- First-crossing error is the first upward 0.5 crossing, interpolated in log-amount, over every case that has a crossing; `Coverage` is the fraction of cases with a crossing. It is not conditioned on the curve having exactly one flip. The bracketed range is the min/max of the per-seed medians, not a confidence interval; for a noisy curve (config A) the first crossing is an early noise crossing.
- `Tolerance step` requires the correct account at every amount outside a ±2% band around the boundary. The old `perfect step` also required the ±0.001% points and is 0% by construction.
- `Non-monotone` means any negative step in p(above) larger than 0.05; `Max neg. step` reports the largest such drop.
- `Real near-boundary` counts actual test items within ±12.5% of a rule boundary (not synthetic sweep points); it is shown per seed.
