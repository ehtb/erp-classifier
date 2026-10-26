# Building and training the ERP classifier

Instructions for training the multimodal classifier, running evaluations, and generating new synthetic invoice data.

## Prerequisites

```bash
# Python 3.10+
pip install -e .

# Browser demo (Gradio)
pip install -e ".[demo]"

# Dev tools and test suite (pytest, ruff, black)
pip install -e ".[dev]"

# AWS Bedrock client (optional, for data generation)
pip install -e ".[aws]"
```

## Quick start: train from the included splits

The repository ships with the filtered source records (`output/03_filtered.jsonl`) and the current grouped split (`output/multimodal/`): 659 original training records plus 644 rule-derived counterfactual copies, 141 validation records, and 141 test records. Training takes about 4 minutes per seed on Apple Silicon and requires no API keys:

```bash
# Train the default model (piecewise-linear encoding + counterfactual weights)
python train.py \
  --train-data output/multimodal/train_multimodal.jsonl \
  --val-data output/multimodal/val_multimodal.jsonl \
  --output-dir models/multimodal-classifier \
  --num-epochs 8

# Evaluate (writes evaluation_summary.json and sweep_curve.png)
python scripts/evaluate.py \
  --model-dir models/multimodal-classifier \
  --val-data output/multimodal/val_multimodal.jsonl \
  --test-data output/multimodal/test_multimodal.jsonl \
  --output-dir output/evaluation-multimodal
```

Model checkpoints live under `models/` (gitignored). Once trained, both `python classify_erp.py` and `python demo/app.py` load `models/multimodal-classifier` by default.

## Generate new synthetic data

`scripts/generate_data.py` supports OpenAI, OpenRouter, and AWS Bedrock. Generating 1,000 samples takes 10 to 20 minutes:

| Backend | Example models | Cost per 1,000 samples | Required environment variables |
|---------|----------------|------------------------|--------------------------------|
| OpenAI | `gpt-4o-mini` ($0.15/$0.60 per 1M), `gpt-4o` ($2.50/$10 per 1M) | ~$0.50 to $2.00 | `OPENAI_API_KEY` |
| OpenRouter | `anthropic/claude-3.5-sonnet` ($3/1M), `meta-llama/llama-3.1-70b-instruct` ($0.35/1M), `google/gemini-pro-1.5` ($0.125/1M) | ~$0.10 to $1.50 | `OPENROUTER_API_KEY` |
| AWS Bedrock | `eu.amazon.nova-pro-v1:0` ($0.60/$2.40 per 1M), `eu.amazon.nova-lite-v1:0` ($0.12/$0.48 per 1M), `anthropic.claude-sonnet-4-5-20250929-v1:0` | ~$0.30 to $1.40 | AWS credentials (`aws configure` or IAM role) + `bedrock:InvokeModel` |

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
python scripts/generate_data.py \
  --llm-backend openai \
  --llm-model gpt-4o-mini \
  --num-samples 1000 \
  --output-dir output/run1

# OpenRouter
export OPENROUTER_API_KEY=sk-or-...
python scripts/generate_data.py \
  --llm-backend openrouter \
  --llm-model anthropic/claude-3.5-sonnet \
  --num-samples 1000 \
  --output-dir output/run1

# AWS Bedrock
export AWS_REGION=us-east-1
python scripts/generate_data.py \
  --llm-backend bedrock \
  --llm-model eu.amazon.nova-pro-v1:0 \
  --num-samples 1000 \
  --output-dir output/run1
```

After generating `03_filtered.jsonl`, build the multimodal dataset and train:

```bash
python scripts/build_dataset.py \
  --input output/run1/03_filtered.jsonl \
  --output-dir output/multimodal

python train.py \
  --train-data output/multimodal/train_multimodal.jsonl \
  --val-data output/multimodal/val_multimodal.jsonl \
  --output-dir models/multimodal-classifier \
  --num-epochs 8
```

## Reproducing the ablations

```bash
# Main A-E grid (writes output/evaluation-multimodal/ablation_summary.md)
python scripts/run_ablation.py --seeds 42 43 44

# Class-weight ablation (writes output/evaluation-multimodal/weight_ablation.md)
python scripts/run_ablation.py --preset weights

# Two-stage model F: D plus a rule-derived price band token (with-prior upper bound).
# Writes its own band_ablation.md (D and F only), so it never overwrites the A-E table.
python scripts/run_ablation.py --preset band --seeds 42 43 44

# All configs in one table, after running both presets above
python scripts/summarize_ablation.py --root output/ablation \
  --output output/evaluation-multimodal/ablation_summary.md

# Side-by-side sweep plot comparing Config D and Config A
python scripts/evaluate.py \
  --model-dir models/ablation/D_ple_quantile_cf/seed42 \
  --compare-model-dir models/ablation/A_periodic_nocf/seed42 \
  --val-data output/multimodal/val_multimodal.jsonl \
  --test-data output/multimodal/test_multimodal.jsonl \
  --output-dir output/evaluation-multimodal \
  --seed 42

# Non-neural baselines (with and without counterfactuals in one file)
python scripts/baselines.py \
  --data-dir output/multimodal \
  --output output/evaluation-multimodal/baselines.json

# Seed ensemble risk-coverage and sweep disagreement
python scripts/evaluate_ensemble.py \
  --model-dirs models/ablation/D_ple_quantile_cf/seed42 \
               models/ablation/D_ple_quantile_cf/seed43 \
               models/ablation/D_ple_quantile_cf/seed44 \
  --test-data output/multimodal/test_multimodal.jsonl \
  --output output/evaluation-multimodal/ensemble.json
```

### Two-stage price band model (Config F)

Stage 1 labels each amount with the price band it falls in. The band edges come from probing the accounting rules (`price_band_edges()` in `src/boundaries.py`), which yields one edge at 800 EUR under the current rules. Stage 2 is the standard classifier, reading `[PRICE_BAND] 0` or `[PRICE_BAND] 1` at the start of the text.

```bash
python scripts/build_dataset.py --input output/03_filtered.jsonl \
  --output-dir output/multimodal-band --price-band
```

The edges are saved as `price_band.json` next to the dataset and copied into the checkpoint by `train.py`. Evaluation, the ensemble script, the baselines, `classify_erp.py`, and `demo/app.py` read that file and recompute the token for every swept or typed amount. Checkpoints without the file behave as single-stage models. Because the edges come from the labelling rule, read Config F's sweep numbers as an upper bound when the boundary is known ahead of time.

## Architecture details

### Model (`ERPClassifierModel` in `src/model.py`)

```
Text Branch (DistilBERT):
  - Input: "[VENDOR] {vendor} [DESC] {description}"
  - Output: 768-dim embedding

Numerical Branch (Piecewise-Linear Encoding + 2-layer MLP):
  - Input: [normalized_amount, log1p(amount)]
  - PLE: 2 raw features + 16 monotone ramps over bin edges fit on train only
  - Output: 64-dim projection

Categorical Branch (Embedding):
  - Input: department_id (0-5, with 0 = UNK)
  - Output: 32-dim embedding

FiLM Gate + Fusion Classifier:
  - gamma, beta = Linear(64 + 32 -> 2 * 768)
  - cls_mod = cls * (1 + tanh(gamma)) + beta
  - Concat(768 + 64 + 32) -> 864-dim -> 256 -> 128 -> 8 (GL accounts)
```

### Training configuration (`train.py`)

```python
optimizer = AdamW(lr=3e-5 encoder, 5e-4 tabular, weight_decay=0.01)
warmup_steps = 15% of total steps
batch_size = 16
epochs = 8
loss = CrossEntropyLoss(label_smoothing=0.05, sample_weight=...)
```

- `--class-weight-mode cf` (default): weights each original IT record and its 4 counterfactual copies by $1/5$ to preserve the source class prior; `balanced` and `none` are kept for the ablation.
- Parameters: ~66.8M (all trainable, ~4 minutes per seed on Apple MPS).

### Data pipeline

Synthetic data generation (`scripts/generate_data.py`):
1. `src/seed_generator.py`: combinatorial specs (vendor, department, language, style, price), oversampling 650 to 950 EUR by 5x.
2. `src/verbalizer.py`: LLM generates product descriptions in EN, DE, and NL.
3. `src/corruptor.py`: applies SAP 40-char truncation (`SAP_SGTXT`), PO prefixes, and OCR noise.
4. `src/critic.py`: zero-shot quality filter (writes rejected items to `03_rejected.jsonl`).

Multimodal dataset builder (`scripts/build_dataset.py`):
- Strips explicit prices from descriptions (`scrub_prices` in `src/text.py`).
- Fits `StandardScaler` on `[net_amount, log1p(net_amount)]` using the training split only.
- Groups near-duplicate product titles (token Jaccard >= 0.5) and assigns whole groups to 70% train, 15% validation, and 15% test.

## Using the trained model in Python

```python
from classify_erp import ERPClassifier

classifier = ERPClassifier("models/multimodal-classifier", confidence_cutoff=0.90)

result = classifier.predict(
    vendor="Amazon Services Europe",
    description="Dell UltraSharp Monitor",
    department="IT_OPS",
    net_amount_eur=750.0,
)
print(result["gl_account"], f"{result['confidence']:.1%}", result["routing"])
```

## Troubleshooting

- **Low critic pass rate (<50%):** switch to a stronger model (`gpt-4o`, `claude-3.5-sonnet`, or `nova-pro`) and inspect `output/03_rejected.jsonl`.
- **Missing API key:** set `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or run `aws configure` before calling `scripts/generate_data.py`.
- **Checkpoint missing preprocessing files:** `classify_erp.py` requires `scaler.json` and `dept_vocab.json` inside the checkpoint directory so a model is never paired with the wrong scaler. Re-run `train.py`, which copies them automatically.

## Customization

- **Add GL accounts:** edit `GLAccount` and `ACCOUNTING_RULES` in `src/config.py`, then update `src/verbalizer.py` and `src/critic.py`. `src/boundaries.py` discovers any new price thresholds automatically.
- **Swap the text encoder:** pass `--model-name microsoft/mdeberta-v3-base` or `--model-name answerdotai/ModernBERT-base` to `train.py`.

## File structure

```
src/
├── config.py              # GL taxonomy, accounting rules, model config
├── boundaries.py          # Rule boundaries, counterfactual and sweep amounts
├── metrics.py             # Wilson/group-bootstrap CIs, sweep metrics
├── text.py                # Explicit price-quote scrubbing and price-band tokens
├── seed_generator.py      # Combinatorial seed generation
├── verbalizer.py          # LLM prompts for product descriptions
├── corruptor.py           # ERP corruption patterns
├── critic.py              # Zero-shot quality filter
├── model.py               # ERPClassifierModel (PLE + 3-branch fusion + FiLM)
├── multimodal_dataset.py  # Grouped split, scrubbing, counterfactuals, weights
└── llm_client.py          # Unified LLM backend (OpenAI, OpenRouter, Bedrock)

Root scripts & directories:
├── classify_erp.py        # CLI inference and counterfactual sweep tool
├── train.py               # Training script
├── demo/
│   └── app.py             # Interactive Gradio browser demo
└── docs/
    ├── architecture.md    # Non-scientific architecture guide with examples and tradeoffs
    └── building.md        # Building, training, and evaluation guide

scripts/ (pipeline tools):
├── generate_data.py       # End-to-end data generation pipeline
├── build_dataset.py       # Convert JSONL -> multimodal splits
├── evaluate.py            # Test evaluation, sweep metrics, and sweep_curve.png / boundary_sweep.png
├── baselines.py           # TF-IDF/GBDT baselines + their amount sweeps
├── run_ablation.py        # Reproduce the A-E, class-weight, and D-vs-F ablations
├── evaluate_ensemble.py   # Seed ensemble risk-coverage and sweep disagreement
└── summarize_ablation.py  # Markdown and CSV aggregation
```

## License

MIT
