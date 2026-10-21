# Building and Training the ERP Classifier

Technical documentation for generating data and training the multimodal ERP classifier.

## Prerequisites

```bash
# Python 3.10+
pip install -e .

# For AWS Bedrock (optional)
pip install -e ".[aws]"
```

## Quick Start: Train from Pre-Generated Data

The repository includes demo data (942 samples). Train a model in ~3 minutes:

```bash
# Data is already included in output/multimodal/
# Just train:
python train.py \
  --train-data output/multimodal/train_multimodal.jsonl \
  --val-data output/multimodal/val_multimodal.jsonl \
  --output-dir models/multimodal-classifier \
  --num-epochs 10

# Evaluate
python scripts/evaluate.py \
  --model-dir models/multimodal-classifier \
  --test-data output/multimodal/test_multimodal.jsonl \
  --output-dir output/evaluation-multimodal
```

**Time**: ~3 minutes (training only)  
**Cost**: $0 (uses existing data)

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
# Build multimodal dataset
python scripts/build_dataset.py \
  --input output/run1/03_filtered.jsonl \
  --output-dir output/multimodal

# Train
python train.py \
  --train-data output/multimodal/train_multimodal.jsonl \
  --val-data output/multimodal/val_multimodal.jsonl \
  --output-dir models/multimodal-classifier \
  --num-epochs 10

# Evaluate
python scripts/evaluate.py \
  --model-dir models/multimodal-classifier \
  --test-data output/multimodal/test_multimodal.jsonl \
  --output-dir output/evaluation-multimodal
```

## Architecture Details

### Model: ERPClassifierModel

Three-branch fusion architecture (src/model.py):

```
Text Branch (DistilBERT):
  - Input: "[VENDOR] {vendor} [DESC] {description}"
  - Output: 768-dim embedding

Numerical Branch (2-layer MLP):
  - Input: [normalized_amount, log1p(amount)]
  - Output: 64-dim projection
  - Learns thresholds (800 EUR) via backpropagation

Categorical Branch (Embedding):
  - Input: department_id (0-5)
  - Output: 32-dim embedding

Fusion Classifier:
  - Concat(768 + 64 + 32) → 864-dim
  - MLP: 864 → 256 → 128 → 8 (GL accounts)
```

**Key innovation**: The numerical branch learns the 800 EUR capitalization threshold natively, rather than trying to parse numbers from text tokens.

### Training Configuration

```python
# From train_multimodal.py
optimizer = AdamW(lr=3e-5, weight_decay=0.01)
warmup_steps = int(total_steps * 0.1)
batch_size = 16
epochs = 10
early_stopping_patience = 5
loss = CrossEntropyLoss()
```

**Parameters**:
- Total: 66.6M (all trainable, full fine-tuning)
- Training time: ~3 minutes (CPU/MPS)

### Data Pipeline

**Synthetic Data Generation** (scripts/generate_data.py):

1. **Seed Generator**: Combinatorial specs (vendor × dept × language × style × price)
   - Oversamples 650-950 EUR range 5× for boundary learning
2. **Verbalizer**: LLM generates realistic product descriptions
3. **Corruptor**: Applies SAP 40-char truncation, PO prefixes, OCR errors
4. **Critic**: Zero-shot quality filter (pass rate depends on LLM)
5. **Multimodal Transform**: Extract text, normalize numerical features, encode departments

**Multimodal Dataset Builder** (scripts/build_dataset.py):

```python
# Rule-free feature extraction
text = f"[VENDOR] {vendor} [DESC] {description}"  # No price/dept in text!

# Continuous features
num_features = [net_amount, log1p(net_amount)]
num_features = (num_features - mean) / std  # StandardScaler (fit on train only)

# Categorical features
dept_id = dept_vocab[department]  # 0=UNK, 1=ENG, 2=FACILITIES, ...

# Stratified split: 70% train, 15% val, 15% test
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
from transformers import AutoTokenizer, AutoConfig
from src.model import ERPClassifierModel
from src.config import ID2LABEL
import torch, numpy as np, json

# Load artifacts
tokenizer = AutoTokenizer.from_pretrained("models/multimodal-classifier")
config = AutoConfig.from_pretrained("models/multimodal-classifier")

with open("output/multimodal/scaler.json") as f:
    scaler = json.load(f)
with open("output/multimodal/dept_vocab.json") as f:
    dept_vocab = json.load(f)['dept2id']

# Load model
model = ERPClassifierModel(config, "distilbert-base-uncased", len(dept_vocab), 2)
model.load_state_dict(torch.load("models/multimodal-classifier/model.safetensors"))
model.eval()

# Predict
def predict(vendor, description, department, net_amount_eur):
    text = f"[VENDOR] {vendor} [DESC] {description}"
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
├── config.py              # GL taxonomy, model config
├── seed_generator.py      # Combinatorial seed generation
├── verbalizer.py          # LLM prompts for product descriptions
├── corruptor.py           # ERP corruption patterns
├── critic.py              # Zero-shot quality filter
├── model.py               # ERPClassifierModel (3-branch fusion)
├── multimodal_dataset.py  # Rule-free feature extraction
└── llm_client.py          # Unified LLM backend (OpenAI, OpenRouter, Bedrock)

Root scripts:
├── classify_erp.py        # CLI inference tool (user-facing)
└── train.py               # Training script

scripts/ (pipeline tools):
├── generate_data.py       # End-to-end data generation pipeline
├── build_dataset.py       # Convert JSONL → multimodal splits
└── evaluate.py            # Test evaluation & metrics
```

## License

MIT
