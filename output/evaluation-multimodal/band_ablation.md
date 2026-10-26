| Config | Seeds | Test acc (seed sd) | Macro F1 | ECE | Real near-boundary |
|---|---|---|---|---|---|
| D_ple_quantile_cf | 3 | 98.8% ± 0.9% | 0.988 | 0.035 | 8/8, 8/8, 8/8 |
| F_ple_quantile_cf_band | 3 | 99.1% ± 0.9% | 0.990 | 0.037 | 8/8, 8/8, 8/8 |

| Config | Sweep point | Strict | Tight ±6.25% | Tolerance step | One flip | Non-monotone | Max neg. step | First-crossing error (seed range) | Coverage |
|---|---|---|---|---|---|---|---|---|---|
| D_ple_quantile_cf | 0.953 | 0.948 | 0.807 | 100% | 100% | 0% | 0.002 | 0.5% [0.4, 0.7] | 100% |
| F_ple_quantile_cf_band | 0.989 | 0.989 | 0.955 | 100% | 100% | 0% | 0.002 | 0.0% [0.0, 0.1] | 100% |

Notes:
- `±` is the training-seed standard deviation on one fixed test set, not test-set uncertainty.
- First-crossing error is the first upward 0.5 crossing, interpolated in log-amount, over every case that has a crossing; `Coverage` is the fraction of cases with a crossing. It is not conditioned on the curve having exactly one flip. The bracketed range is the min/max of the per-seed medians, not a confidence interval; for a noisy curve (config A) the first crossing is an early noise crossing.
- `Tolerance step` requires the correct account at every amount outside a ±2% band around the boundary. The old `perfect step` also required the ±0.001% points and is 0% by construction.
- `Non-monotone` means any negative step in p(above) larger than 0.05; `Max neg. step` reports the largest such drop.
- `Real near-boundary` counts actual test items within ±12.5% of a rule boundary (not synthetic sweep points); it is shown per seed.
- Two-stage price band model (`F_ple_quantile_cf_band`): the band edges come from the same accounting rule that produces the labels, so the sweep and crossing numbers are a with-prior upper bound, not evidence that the boundary was learned. The `price_stripped` text ablation does not remove the band token.
