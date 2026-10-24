# Building and Training the ERP Classifier

Technical documentation for generating data and training the multimodal ERP classifier.

## Prerequisites

```bash
# Python 3.10+
pip install -e .

# Dev tools and test suite (pytest, ruff, black)
pip install -e ".[dev]"

# For AWS Bedrock (optional)
pip install -e ".[aws]"
```

## Quick Start: Train from Pre-Generated Data

The repository includes the source records (`output/03_filtered.jsonl`) and the current demo split (`output/multimodal/`). The split is grouped and stratified: 659 original training records plus 644 rule-derived counterfactual copies, 141 validation, 141 test. Training takes roughly 4 minutes per seed on Apple Silicon:

```bash
# Data is already included in output/multimodal/
python train.py \
  --train-data output/multimodal/train_multimodal.jsonl \
  --val-data output/multimodal/val_multimodal.jsonl \
  --output-dir models/multimodal-classifier \
  --num-epochs 8

# Evaluate
python scripts/evaluate.py \
  --model-dir models/multimodal-classifier \
  --val-data output/multimodal/val_multimodal.jsonl \
  --test-data output/multimodal/test_multimodal.jsonl \
  --output-dir output/evaluation-multimodal
```

**Time**: ~5 minutes (training only)
**Cost**: $0 (uses existing data)

Weights are not committed (`models/` is gitignored). The quickstart above trains them locally.

## Generate New Training Data

### LLM Backend Options

The data generation pipeline supports multiple LLM providers. Choose based on your needs:

| Backend | Best For | Cost (1000 samples) | Setup |
|---------|----------|---------------------|-------|
| **OpenAI** | Best quality | ~$0.50-2.00 | API key |
| **OpenRouter** | Model variety, competitive pricing | ~$0.10-1.50 | API key |
| **Bedrock** | AWS integration, Nova models | ~$1.40 | AWS credentials |

#### Option 1: OpenAI (Recommended for Production)

```bash
# Set API key
export OPENAI_API_KEY=sk-...
export LLM_BACKEND=openai
export LLM_MODEL=gpt-4o-mini  # or gpt-4o for best quality

# Generate data
python scripts/generate_data.py \
  --llm-backend openai \
  --llm-model gpt-4o-mini \
  --num-samples 1000 \
  --output-dir output/run1
```

**Cost**: ~$0.50-2.00 for 1000 samples  
**Quality**: Excellent (95%+ critic pass rate)  
**Speed**: ~10-15 minutes

**Model options**:
- `gpt-4o`: Best quality ($2.50/1M input, $10/1M output)
- `gpt-4o-mini`: Fast and cheap ($0.15/1M input, $0.60/1M output) ← **Recommended**
- `gpt-3.5-turbo`: Legacy ($0.50/1M input, $1.50/1M output)

#### Option 2: OpenRouter (Best Value)

```bash
# Get API key from https://openrouter.ai/keys
export OPENROUTER_API_KEY=sk-or-...
export LLM_BACKEND=openrouter
export LLM_MODEL=anthropic/claude-3.5-sonnet  # or others

# Generate data
python scripts/generate_data.py \
  --llm-backend openrouter \
  --llm-model anthropic/claude-3.5-sonnet \
  --num-samples 1000 \
  --output-dir output/run1
```

**Cost**: ~$0.10-1.50 for 1000 samples (model-dependent)  
**Quality**: Varies by model  
**Speed**: ~10-20 minutes

**Recommended models**:
- `anthropic/claude-3.5-sonnet`: Best quality ($3/1M tokens)
- `meta-llama/llama-3.1-70b-instruct`: Good quality, fast ($0.35/1M)
- `google/gemini-pro-1.5`: Cheap, decent quality ($0.125/1M)
- `mistralai/mistral-large`: Fast, good quality ($2/1M)

See all models: https://openrouter.ai/models

#### Option 3: AWS Bedrock (AWS Integration)

```bash
# Configure AWS credentials
aws configure

# Or use IAM role / environment variables
export AWS_REGION=us-east-1
export LLM_BACKEND=bedrock
export LLM_MODEL=eu.amazon.nova-pro-v1:0

# Generate data
python scripts/generate_data.py \
  --llm-backend bedrock \
  --llm-model eu.amazon.nova-pro-v1:0 \
  --num-samples 1000 \
  --output-dir output/run1
```

**Cost**: ~$1.40 for 1000 samples (Nova Pro)  
**Quality**: Excellent (94%+ critic pass rate)  
**Speed**: ~15 minutes

**Prerequisites**:
- AWS account with Bedrock access
- Enable model access in AWS Console → Bedrock → Model access
- IAM permissions: `bedrock:InvokeModel`

**Model options**:
- `eu.amazon.nova-pro-v1:0`: Fast, cheap ($0.60/1M input, $2.40/1M output) ← **Recommended**
- `eu.amazon.nova-lite-v1:0`: Ultra cheap ($0.12/1M input, $0.48/1M output)
- `anthropic.claude-sonnet-4-5-20250929-v1:0`: Best quality ($3/1M input, $15/1M output)
- `meta.llama3-70b-instruct-v1:0`: Good balance ($1.35/1M input, $4.05/1M output)

### After Data Generation

Once you have filtered data, build the multimodal dataset and train:

```bash
# Build multimodal dataset (grouped split, price scrubbing, 4 counterfactuals/IT record)
python scripts/build_dataset.py \
  --input output/run1/03_filtered.jsonl \
  --output-dir output/multimodal

# Train (default --class-weight-mode cf preserves the source class prior)
python train.py \
  --train-data output/multimodal/train_multimodal.jsonl \
  --val-data output/multimodal/val_multimodal.jsonl \
  --output-dir models/multimodal-classifier \
  --num-epochs 8

# Evaluate
python scripts/evaluate.py \
  --model-dir models/multimodal-classifier \
  --val-data output/multimodal/val_multimodal.jsonl \
  --test-data output/multimodal/test_multimodal.jsonl \
  --output-dir output/evaluation-multimodal
```

## Reproducing the Ablations

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

### Two-stage price band model (config F)

Stage 1 labels each amount with the price band it falls in. The band edges are
found by probing the accounting rules (`price_band_edges()` in `src/boundaries.py`),
so there is one edge today, at 800 EUR. Stage 2 is the usual classifier, which reads
the label as a `[PRICE_BAND] 0` or `[PRICE_BAND] 1` token at the start of the text.

```bash
# Minimal example; config F also uses --counterfactuals-per-record 4 (the default)
# and --bin-strategy quantile, as in scripts/run_ablation.py.
python scripts/build_dataset.py --input output/03_filtered.jsonl \
  --output-dir output/multimodal-band --price-band
```

The edges are saved as `price_band.json` next to the dataset and are copied into the
checkpoint by `train.py`. Evaluation, the ensemble script, the baselines and
`classify_erp.py` read that file and recompute the token for every swept or typed
amount. Older checkpoints without the file behave as before.

Because the edges come from the rule that produces the labels, this setup hands
the model the boundary. Read its sweep numbers as an upper bound, not as evidence
that the boundary was learned.

## Architecture Details

### Model: ERPClassifierModel

Three-branch fusion architecture (`src/model.py`):

```
Text Branch (DistilBERT):
  - Input: "[VENDOR] {vendor} [DESC] {description}"
  - Output: 768-dim embedding

Numerical Branch (Piecewise-Linear Encoding + 2-layer MLP):
  - Input: [normalized_amount, log1p(amount)]
  - PLE: 2 raw features + 16 monotone ramps over bin edges fit on train only
  - Output: 64-dim projection

Categorical Branch (Embedding):
  - Input: department_id (0-5)
  - Output: 32-dim embedding

Fusion Classifier:
  - Concat(768 + 64 + 32) → 864-dim
  - MLP: 864 → 256 → 128 → 8 (GL accounts)
```

### Training Configuration

```python
optimizer = AdamW(lr=3e-5 encoder, 5e-4 tabular, weight_decay=0.01)
warmup_steps = 15% of total steps
batch_size = 16
epochs = 8
loss = CrossEntropyLoss(label_smoothing=0.05, sample_weight=...)
```

- `--class-weight-mode cf` (default): weights each original IT record and its 4 counterfactual copies by $1/5$ to preserve the source class prior; `balanced` and `none` are kept for the ablation.
- **Parameters**: ~66.8M (all trainable, ~4 minutes per seed on Apple MPS).

### Data Pipeline

**Synthetic Data Generation** (`scripts/generate_data.py`):

1. **Seed Generator**: Combinatorial specs (vendor × dept × language × style × price), oversampling 650–950 EUR 5×
2. **Verbalizer**: LLM generates realistic product descriptions
3. **Corruptor**: Applies SAP 40-char truncation, PO prefixes, OCR errors
4. **Critic**: Zero-shot quality filter
5. **Multimodal Transform**: Extract text, normalize numerical features, encode departments

**Multimodal Dataset Builder** (`scripts/build_dataset.py`):

```python
# Text: vendor + description, with explicit price quotes removed at build time
text = f"[VENDOR] {vendor} [DESC] {scrub_prices(description)}"

# Continuous features (StandardScaler fit on train only)
num_features = ([net_amount, log1p(net_amount)] - mean) / std

# Categorical features
dept_id = dept_vocab[department]  # 0=UNK, 1=ENG, 2=FACILITIES, ...

# Split: 70% train, 15% val, 15% test (greedy whole-group assignment by title Jaccard)
```

## Using the Trained Model

### Command-Line Tool

```bash
# Interactive mode
python classify_erp.py

# One-shot prediction
python classify_erp.py \
  --vendor "Amazon" \
  --description "Dell Monitor" \
  --department IT_OPS \
  --amount 450
```

See main [README](README.md) for detailed usage.

### Direct Python Integration

```python
from transformers import AutoTokenizer
from src.model import ERPClassifierModel
from src.config import ID2LABEL
from src.text import load_price_band_edges, scrub_prices, set_price_band
import torch, numpy as np, json

MODEL_DIR = "models/multimodal-classifier"

# train.py copies scaler.json, dept_vocab.json, numeric_bins.json (and price_band.json for
# a two-stage checkpoint) next to the checkpoint; from_pretrained restores num_encoding and the bin edges from config.
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = ERPClassifierModel.from_pretrained(MODEL_DIR).eval()

with open(f"{MODEL_DIR}/scaler.json") as f:
    scaler = json.load(f)
with open(f"{MODEL_DIR}/dept_vocab.json") as f:
    dept_vocab = json.load(f)['dept2id']

# Only two-stage (--price-band) checkpoints ship price_band.json; for those the band
# token must be added for every amount, or the model is silently misconfigured.
band_edges = load_price_band_edges(MODEL_DIR)  # None for a normal checkpoint

# Predict
def predict(vendor, description, department, net_amount_eur):
    text = f"[VENDOR] {vendor} [DESC] {scrub_prices(description)}"
    if band_edges is not None:
        text = set_price_band(text, net_amount_eur, band_edges)
    encoding = tokenizer(text, truncation=True, max_length=128, 
                         padding='max_length', return_tensors='pt')
    
    num_features = np.array([net_amount_eur, np.log1p(net_amount_eur)])
    num_features = (num_features - np.array(scaler['mean'])) / np.array(scaler['scale'])
    
    dept_id = dept_vocab.get(department, 0)
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            num_features=torch.from_numpy(num_features).unsqueeze(0).float(),
            dept_id=torch.tensor([dept_id], dtype=torch.long),
        )
        probs = torch.softmax(outputs['logits'], dim=-1)
        pred_idx = torch.argmax(probs).item()
    
    return ID2LABEL[pred_idx], float(probs[0, pred_idx])

# Example
gl_account, confidence = predict(
    "Amazon Services", 
    "Dell Monitor", 
    "IT_OPS", 
    450.0
)
print(f"{gl_account} ({confidence:.1%})")
```

## Cost Comparison

Data generation cost for 1,000 samples:

| Backend | Model | Cost | Quality | Speed |
|---------|-------|------|---------|-------|
| OpenAI | gpt-4o-mini | ~$0.50 | Excellent | 10-15 min |
| OpenRouter | Claude 3.5 Sonnet | ~$1.50 | Excellent | 10-15 min |
| OpenRouter | Llama 3.1 70B | ~$0.35 | Good | 15-20 min |
| OpenRouter | Gemini Pro 1.5 | ~$0.10 | Decent | 15-20 min |
| Bedrock | Nova Pro | ~$1.40 | Excellent | 15 min |
| Bedrock | Nova Lite | ~$0.30 | Good | 15 min |

## Troubleshooting

**Low Critic pass rate (<50%)**:
- Try a better model (GPT-4o, Claude, Nova Pro)
- Check `output/03_rejected.jsonl` for patterns
- Verify prompts in `src/verbalizer.py` include concrete examples

**"API key not found"**:
- Set appropriate environment variable:
  - OpenAI: `OPENAI_API_KEY`
  - OpenRouter: `OPENROUTER_API_KEY`
  - Bedrock: Use `aws configure`

**Poor model performance on specific GL class**:
- Inspect class distribution in `output/multimodal/`
- Check if category prompts generate clear semantic signals

**Dependency issues**:
- Run `pip install -e .`
- For NumPy/PyArrow conflicts: `numpy<2`, `pyarrow<16`, `datasets==2.21.0`

## Customization

**Add GL accounts**: Edit `GLAccount` enum in `src/config.py`, update verbalizer templates and critic prompt.

**Change model**: Set `model_name` in training script:
```python
model_name = "answerdotai/ModernBERT-base"  # or "microsoft/mdeberta-v3-base"
```

**Adjust sampling**: Edit `src/config.py`:
```python
DATA_GEN_CONFIG = {
    "threshold_oversample_factor": 5,  # 5x sampling near 800 EUR
    "corruption_rate": 0.3,
    "train_ratio": 0.7,
}
```

## File Structure

```
src/
├── config.py              # GL taxonomy, accounting rules, model config
├── boundaries.py          # Rule boundaries, counterfactual and sweep amounts
├── metrics.py             # Wilson/group-bootstrap CIs, sweep metrics
├── text.py                # Explicit price-quote scrubbing
├── seed_generator.py      # Combinatorial seed generation
├── verbalizer.py          # LLM prompts for product descriptions
├── corruptor.py           # ERP corruption patterns
├── critic.py              # Zero-shot quality filter
├── model.py               # ERPClassifierModel (PLE + 3-branch fusion)
├── multimodal_dataset.py  # Grouped split, scrubbing, counterfactuals, weights
└── llm_client.py          # Unified LLM backend (OpenAI, OpenRouter, Bedrock)

Root scripts:
├── classify_erp.py        # CLI inference tool (user-facing)
└── train.py               # Training script

scripts/ (pipeline tools):
├── generate_data.py       # End-to-end data generation pipeline
├── build_dataset.py       # Convert JSONL → multimodal splits
├── evaluate.py            # Test evaluation, sweep, text ablations
├── baselines.py           # TF-IDF/GBDT baselines + their amount sweeps
├── run_ablation.py        # Reproduce the A-E, class-weight and D-vs-F (price band) ablations
├── evaluate_ensemble.py   # Seed ensemble risk-coverage and sweep disagreement
└── summarize_ablation.py  # Markdown aggregation
```

## License

MIT
