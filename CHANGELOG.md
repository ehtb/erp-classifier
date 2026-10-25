# Changelog

All notable changes to the ERP Classifier project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.5.0] - 2026-10-25

**In short:** We tried giving the model a head start on the 800 EUR line: a first step labels each price as "below" or "above" 800, and the classifier reads that label along with the description. The switch from operating expense to capital expense now lands on 800 (to within the sweep's grid resolution) instead of about 0.6% away, while overall accuracy stays within noise (95.7% vs 96.5%). Because the first step uses the same rule that creates the labels, this shows the upper bound when the boundary is known ahead of time. We also added a CLI `--sweep` table, confidence routing output, a side-by-side boundary sweep plot, and a Gradio browser demo (`app.py`).

### Added
- **Two-stage price band model** (`src/boundaries.py`, `src/text.py`, `src/multimodal_dataset.py`). `scripts/build_dataset.py --price-band` labels each amount with its band, using edges found by probing the accounting rules, and prepends `[PRICE_BAND] n` to the text. Counterfactual copies get the band for their own amount. The edges are saved as `price_band.json` and copied into the checkpoint.
- **Amount-aware inference and sweeps** (`scripts/evaluate.py`, `scripts/evaluate_ensemble.py`, `scripts/baselines.py`, `classify_erp.py`). The band token is recomputed for every swept or typed amount, and checkpoints without `price_band.json` behave as before.
- **Config F** in `scripts/run_ablation.py` (`--preset band` compares it with D), and `tests/test_price_band.py`.
- **Interactive demo and sweep visuals** (`classify_erp.py`, `app.py`, `scripts/evaluate.py`). `classify_erp.py` now supports `--sweep` for counterfactual price tables, displays confidence routing (`AUTO-POST` vs `MANUAL REVIEW` at `--cutoff 0.90`), and reports FiLM modulation stats under `--show-all`. `scripts/evaluate.py` saves `sweep_curve.png` for a single model or `boundary_sweep.png` side-by-side when `--compare-model-dir` is given. `app.py` provides a Gradio web interface with an amount slider and live sweep chart.

### Results (3 seeds, same split as D)
| Config | Test acc | Sweep point acc | Tolerance step | One flip | Non-monotone | First crossing |
|---|---|---|---|---|---|---|
| D PLE + CF | 96.5% ± 0.6% | 0.950 | 100% | 100% | 0% | 0.6% [0.5, 0.7] |
| F D + price band token | 95.7% ± 1.0% | 1.000 | 100% | 100% | 0% | 0.0006% |

- F puts the flip on the boundary, and the sweep is right at every point. Test accuracy is within noise (the 0.8-point gap is about one item).
- The constant-text check, which blanks the description, behaves like D (IT items stay at 93.9%), so the token adds nothing beyond what amount and department already give.
- Because the edge comes from the labelling rule, F is a with-prior upper bound.

## [0.4.0] - 2026-10-24

**In short:** The model now handles the 800 EUR line much more cleanly: for the same product, it switches from operating expense to capital expense at the right price and stays there, instead of wobbling around it. We also corrected an earlier number that made the old version look better than it was, and made the test set stricter. Accuracy is 96.5%, the same as a much simpler word-count-plus-price-bucket model, so the reason to use the neural model is the smooth price behaviour, not the headline score.

### Added & Changed
- **Piecewise-linear amount encoding** (`src/model.py`). The default amount encoder is now `PiecewiseLinearEncoding` (`--num-encoding ple`), replacing `PeriodicNumericalEncoding`. It builds ramps over `log1p(amount)` between bin edges fit on the training data, so a higher amount can only push the score one way. Options: `--bin-strategy quantile|tree` and `--bin-weighting unweighted|weighted`.
- **Boundaries and counterfactual copies** (`src/boundaries.py`, `src/multimodal_dataset.py`). Boundaries are found by probing `get_gl_account()`, so no threshold is hard-coded. Each amount-sensitive training record gets 4 copies with the amount moved near the boundary and the label recomputed. An original and its k copies each weigh 1/(k+1), which keeps the class mix unchanged (`--class-weight-mode cf`).
- **Grouped splits and price scrubbing** (`src/multimodal_dataset.py`, `src/text.py`). Records with overlapping titles (token Jaccard of 0.5 or more) stay in the same split, and groups are assigned greedily to even out the labels. The result is 659 original training records plus 644 copies, 141 validation and 141 test records (87 groups in test). Prices written into descriptions are removed at build time and at inference.
- **Honest sweep metrics** (`src/metrics.py`, `scripts/evaluate.py`, `scripts/evaluate_ensemble.py`, `scripts/baselines.py`). The crossing error is now measured on every case, not only the ones with exactly one flip, and it reports how many cases had a crossing at all. Also new: tolerance-step accuracy (correct everywhere outside a ±2% band), strict sweep accuracy, and group-bootstrap confidence intervals.
- **Fewer scripts.** The ablation runners are merged into `scripts/run_ablation.py` (`--preset main|weights`), and the old `src/evaluate.py` is gone.

### Fixed
- **The old crossing number was misleading.** The 0.4% figure for the periodic, no-counterfactual setup in v0.3.0 came from the 1% of cases that had a single flip. Measured on all cases, that setup is off by 15.2% and non-monotone in every case.
- **Rule parsing** (`src/config.py`). An unreadable `threshold_rule` string now raises a `ValueError`. Before, it silently matched every amount.
- **Checkpoints and CLI** (`src/model.py`, `train.py`, `classify_erp.py`). The label count and label names are saved properly. `class_weights` is always registered, which removes a warning when loading. A checkpoint must now ship its own preprocessing files, with no silent fallback. `--amount 0` works in one-shot mode, and negative amounts are clamped before `log1p`.

### Results Summary
- **Default setup** (`D_ple_quantile_cf`, 3 seeds): 96.5% ± 0.6% test accuracy (seed 42: 97.2%). All 8 items near the boundary are correct. The sweep flips once in 100% of cases, is non-monotone in 0%, is right everywhere outside ±2% in 100%, and the first crossing is 0.6% off (range 0.5% to 0.7% across seeds).
- **Class weights.** Balanced class weights boosted the counterfactual-heavy IT classes about 5x. On config `C`, balanced weights give 39% non-monotone cases on seed 42 (94.3% accuracy). With `cf` weights, `C` is 6% non-monotone on seed 42 and 4% across 3 seeds (95.7% ± 1.5%). On `D`, `cf` and `none` are tied within noise: `cf` gets 96.5% ± 0.6% with a 0.6% crossing (range 0.5% to 0.7%), and `none` gets 96.9% ± 0.3% with a 0.6% crossing (range 0.2% to 1.1%). `none` is steadier on accuracy, `cf` is steadier on the crossing and keeps the source class mix.
- **Full tables and baseline sweeps:** [`output/evaluation-multimodal/ablation_summary.md`](output/evaluation-multimodal/ablation_summary.md), [`output/evaluation-multimodal/weight_ablation.md`](output/evaluation-multimodal/weight_ablation.md), [`output/evaluation-multimodal/baselines.json`](output/evaluation-multimodal/baselines.json).

## [0.3.0] - 2026-10-23

**In short:** The model got better at reading amounts and learned when to hold back. Accuracy rose from 93.6% to 97.9% on validation, and with a 0.90 confidence cutoff it posted about 94% of the test invoices on its own with no mistakes, sending the rest to a person.

### Added & Changed
- **Periodic numerical encoding** (`src/model.py`). Added `PeriodicNumericalEncoding` with 16 log-spaced learnable frequency parameters (0.1 to 100) expanding `[z_score(net_amount), log1p(net_amount)]` into a 66-dim `[raw, sin, cos]` vector before the 2-layer MLP projection.
- **FiLM gating** (`src/model.py`). Added a linear projection (`Linear(96, 1536)`) from the concatenated numerical and department embeddings to modulate the DistilBERT `[CLS]` vector via `cls * (1 + tanh(gamma)) + beta`.
- **Modality dropout and label smoothing** (`src/model.py`). Masked `dept_id` to `0 (UNK)` on 15% of training examples so the `UNK` embedding receives gradient updates, and set `label_smoothing=0.05` in `CrossEntropyLoss` to improve confidence calibration.
- **Training schedule** (`train.py`). Reduced epochs from 15 to 8, lowered `--tabular-lr` from `1e-3` to `5e-4`, and increased warmup from 10% to 15% of total steps.
- **Confidence routing evaluation** (`src/evaluate.py`, later replaced by `scripts/evaluate.py` in v0.4.0). Added confusion matrix output, error breakdowns, and coverage/accuracy tables across confidence thresholds.

### Results (v0.3.0 split)
- **Validation (141 samples):** 97.87% accuracy, 0.9783 macro F1 (3 errors, best checkpoint at epoch 5).
- **Test (142 samples):** 97.18% accuracy, 0.9701 macro F1 (4 errors).
- **Error pattern:** All 7 misclassifications across validation and test were €0 net-amount records (credit notes or adjustments) with `UNK` department and low confidence (0.49 to 0.79). At the 0.90 threshold, 133 of 142 test items (93.7%) were auto-posted at 100% accuracy, and the remaining 9 went to manual review.

## [0.2.0] - 2026-10-23

**In short:** A bug-fixing release. Five problems were stopping the model from training well or from loading back correctly after being saved. Nothing new was added.

### Fixed
- **Trainable `UNK` department embedding** (`src/model.py`). Removed `padding_idx=0` from `nn.Embedding(num_depts, 32)`, which had frozen gradients for index 0 (`UNK`).
- **PreTrainedModel serialization** (`src/model.py`). Saved `base_model_name`, `num_depts`, and `num_continuous` in `config`, and renamed `_init_weights` to `_init_custom_heads` to avoid colliding with the HuggingFace `PreTrainedModel` hook.
- **Class weights** (`src/model.py`, `train.py`). Added balanced class weights capped at 5.0 and registered them as a model buffer.
- **Numerical branch activation order** (`src/model.py`). Moved `LayerNorm(64)` after `Linear -> GELU -> Dropout -> Linear -> GELU` with Kaiming initialization. Applying `LayerNorm` before the first nonlinearity had normalized away magnitude differences across the 2D numerical input.
- **Differential learning rates and weight decay** (`train.py`). Set `3e-5` on DistilBERT and `1e-3` on the tabular/fusion heads, with `0.0` weight decay on biases and `LayerNorm` weights. Switched checkpoint selection from `weighted_f1` to `macro_f1`.

### Results
- Validation accuracy improved from 92.91% (epoch 10) to 93.62% (epoch 2, macro F1 0.9353).

## [0.1.0] - 2026-10-23

**In short:** The first working version. It reads the invoice text, the amount, and the department together and picks one of eight GL accounts, with about 93% accuracy on validation.

### Added
- **Three-branch multimodal model** (`src/model.py`). DistilBERT text encoder (768-d), numerical MLP on `[z_score(net_amount), log1p(net_amount)]` (64-d), and department embedding (32-d) concatenated into an 864 -> 256 -> 128 -> 8 classifier head.
- **Training and synthetic data pipeline** (`train.py`, `src/config.py`, `scripts/generate_data.py`). 8 GL account classes, department vocabulary, combinatorial seed generator, LLM verbalizer, SAP text corruptor, and zero-shot critic.
- **Baseline validation accuracy:** 92.91% (weighted F1 0.9268) at epoch 10 of 15.
