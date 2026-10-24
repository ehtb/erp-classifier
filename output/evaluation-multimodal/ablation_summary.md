| Config | Seeds | Test acc (seed sd) | Macro F1 | ECE | Real near-boundary |
|---|---|---|---|---|---|
| A_periodic_nocf | 3 | 94.3% ± 1.2% | 0.942 | 0.053 | 6/8, 8/8, 7/8 |
| B_ple_quantile_nocf | 3 | 94.8% ± 2.0% | 0.948 | 0.035 | 7/8, 6/8, 5/8 |
| C_periodic_cf | 3 | 95.7% ± 1.5% | 0.957 | 0.037 | 8/8, 8/8, 8/8 |
| D_ple_quantile_cf | 3 | 96.5% ± 0.6% | 0.965 | 0.046 | 8/8, 8/8, 8/8 |
| E_ple_tree_cf | 3 | 95.3% ± 0.7% | 0.953 | 0.041 | 8/8, 8/8, 8/8 |

| Config | Sweep point | Strict | Tight ±6.25% | Tolerance step | One flip | Non-monotone | Max neg. step | First-crossing error (seed range) | Coverage |
|---|---|---|---|---|---|---|---|---|---|
| A_periodic_nocf | 0.865 | 0.844 | 0.705 | 14% | 2% | 100% | 0.869 | 15.2% [2.9, 38.9] | 100% |
| B_ple_quantile_nocf | 0.760 | 0.756 | 0.527 | 5% | 87% | 0% | 0.000 | 15.1% [13.5, 16.0] | 87% |
| C_periodic_cf | 0.971 | 0.955 | 0.881 | 99% | 99% | 4% | 0.119 | 0.1% [0.0, 0.1] | 100% |
| D_ple_quantile_cf | 0.950 | 0.948 | 0.795 | 100% | 100% | 0% | 0.006 | 0.6% [0.5, 0.7] | 100% |
| E_ple_tree_cf | 0.935 | 0.935 | 0.731 | 73% | 100% | 0% | 0.000 | 1.6% [0.6, 2.1] | 100% |

Notes:
- `±` is the training-seed standard deviation on one fixed test set, not test-set uncertainty.
- First-crossing error is the first upward 0.5 crossing, interpolated in log-amount, over every case that has a crossing; `Coverage` is the fraction of cases with a crossing. It is not conditioned on the curve having exactly one flip. The bracketed range is the min/max of the per-seed medians, not a confidence interval; for a noisy curve (config A) the first crossing is an early noise crossing.
- `Tolerance step` requires the correct account at every amount outside a ±2% band around the boundary. The old `perfect step` also required the ±0.001% points and is 0% by construction.
- `Non-monotone` means any negative step in p(above) larger than 0.05; `Max neg. step` reports the largest such drop.
- `Real near-boundary` counts actual test items within ±12.5% of a rule boundary (not synthetic sweep points); it is shown per seed.
