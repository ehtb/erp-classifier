# ERP GL Account Classifier

A multimodal ML classifier that predicts GL accounts for ERP transactions by learning from product descriptions, transaction amounts, and department codes together.

**Demo**: [🤗 HuggingFace Dataset](https://huggingface.co/datasets/EHTB/erp-classifier-dataset) | [🤖 Trained Model](https://huggingface.co/EHTB/erp-classifier)

## The Problem

Amazon Business invoices all show the same vendor name. A €750 monitor and a pack of coffee pods both say "Amazon Services Europe" — the model has to look at department, amount, and description together to classify them correctly.

IT hardware has a special challenge: it splits across two GL accounts based on price alone:
- **< 800 EUR** → GL_6220 (operating expense)
- **≥ 800 EUR** → GL_1400 (capital expenditure)

A €750 monitor and an €850 monitor are the same item, different accounting treatment. Text-only classifiers fail here because they see "[NET_AMT] 799.00" as text tokens, not numbers—they can't learn mathematical thresholds.

## The Solution: Multimodal Architecture

Instead of forcing text models to parse numbers, we treat ERP transactions as **hybrid tabular-text data**:

```
┌─────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
│  Text Features          │      │ Categorical Features     │      │ Continuous Features      │
│  vendor + description   │      │ department               │      │ [net_amount, log(amt)]   │
└───────────┬─────────────┘      └──────────┬───────────────┘      └───────────┬──────────────┘
            │                               │                                   │
            ▼                               ▼                                   ▼
┌─────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
│ DistilBERT Encoder      │      │ Embedding Layer          │      │ Numerical Projection     │
│ (pretrained)            │      │ (learned)                │      │ (2-layer MLP)            │
└───────────┬─────────────┘      └──────────┬───────────────┘      └───────────┬──────────────┘
            │ 768-dim                        │ 32-dim                           │ 64-dim
            └──────────────────┬─────────────┴──────────────────────────────────┘
                               ▼
                ┌──────────────────────────────┐
                │   Fusion Classifier          │
                │   Concat → MLP → 8 GL codes  │
                └──────────────────────────────┘
```

The 64-dimensional numerical projection branch learns "< 800 vs ≥ 800" directly through backpropagation, not text parsing.

## Results

**Proof-of-Concept Performance** (942 training samples, ~$1.40 data generation cost):

- **Test Accuracy**: 90.8%
- **Test F1**: 90.6%
- **800 EUR Threshold**: 73.5% accuracy on items < 800 EUR

| GL Account | F1 Score | Confidence |
|------------|----------|------------|
| GL_6410_FACILITY_PANTRY | 100% | Very High |
| GL_6450_CLEANING_MAINTENANCE | 100% | Very High |
| GL_6400_OFFICE_CONSUMABLES | 96.8% | High |
| GL_6250_TECH_BOOKS_TRAINING | 95.0% | High |
| GL_6150_MARKETING_PROMO | 94.1% | High |
| GL_6310_WORKPLACE_SAFETY_ERGONOMICS | 91.9% | High |
| GL_1400_IT_HARDWARE_CAPEX | 79.1% | Moderate |
| GL_6220_IT_PERIPHERALS_OPEX | 64.0% | Low |

**What "High Confidence" Means**: When the model sees clear examples (e.g., "Logitech Mouse, 15 EUR"), it predicts with 97%+ confidence. The 90.8% overall accuracy represents the model's performance across all test samples, including edge cases.

## Limitations

This is a **proof-of-concept** with intentional constraints:

1. **Small training set**: 942 samples (< 1,000), generated in one LLM run
   - Some GL classes have fewer than 100 training examples
   - Boundary cases (IT hardware near 800 EUR) need more examples

2. **Synthetic data quality**: Occasionally creates ambiguous descriptions
   - Example: "beurer 0483" (cryptic code) → wrong prediction (83.9% confidence)
   - Example: "Beurer Blood Pressure Monitor" (descriptive) → correct (97.1% confidence)

3. **Distribution shift**: Model struggles with inputs that differ from training data
   - Trained on full product names ("Dell UltraSharp Monitor")
   - Fails on cryptic ERP codes ("DELL-U27-0483")

4. **Missing categories**: 8 GL accounts don't cover all real-world purchases
   - Health equipment, travel expenses, subscriptions → forced into closest match

## Training Methodology

This POC uses **full fine-tuning** rather than parameter-efficient methods like LoRA (Low-Rank Adaptation).

### Full Fine-Tuning vs LoRA

| Approach | Parameters Trained | Training Time | Model Size | Best For |
|----------|-------------------|---------------|------------|----------|
| **Full Fine-Tuning** | All 66.6M | ~3 min | 255 MB | Small datasets (<5K), multimodal models, max accuracy |
| **LoRA** | ~0.6M (1%) | ~2 min | 8 MB adapters | Large datasets (>10K), text-only, resource constraints |

### Why Full Fine-Tuning for This POC

**1. Small Dataset (942 samples)**
- LoRA adapters need larger datasets to learn effectively
- Full fine-tuning extracts maximum signal from limited data
- Training time difference is negligible at this scale (3 min vs 2 min)

**2. Multimodal Architecture**
- The model fuses three branches: text (DistilBERT), numerical (MLP), categorical (embeddings)
- LoRA typically targets only attention layers in the text encoder
- Full fine-tuning trains the entire fusion network end-to-end

**3. Numerical Threshold Learning**
- The 800 EUR threshold is learned via backpropagation in the numerical branch
- This requires training the MLP projection layers, not just text adapters
- LoRA would freeze these layers, preventing threshold learning

**4. POC Constraints**
- Training happens once, not continuously
- Model deployment is local CLI, not cloud API with strict size limits
- 255 MB model size is acceptable for a desktop application

### When LoRA Makes Sense

For production scaling, LoRA becomes advantageous when:
- Dataset grows to 10,000+ samples (LoRA prevents overfitting)
- Model is deployed as a cloud API (8 MB vs 255 MB reduces latency)
- Continuous retraining with new data (LoRA is faster for iteration)
- Text-only classification (no multimodal fusion network)

The trained model already includes the full weights, so switching to LoRA later is straightforward by training adapters on top of the frozen base.

## Path to Production (>98% Accuracy)

To move from 90.8% → >98% accuracy:

### 1. Scale Training Data (5-10× more samples)
- **Goal**: 5,000-10,000 samples
- **Focus**: More boundary examples (700-900 EUR range), underrepresented classes
- **Cost**: ~$10-20 (LLM API costs), ~2 hours
- **Impact**: Reduces variance, improves boundary precision

### 2. Add Contrastive Examples
Generate twin records with identical descriptions but different amounts or departments:
- "Dell Monitor, IT_OPS, 750 EUR" → GL_6220_OPEX
- "Dell Monitor, IT_OPS, 850 EUR" → GL_1400_CAPEX

Forces the model to use numerical features, not just text patterns.

### 3. Real-World Data Enrichment
For cryptic ERP codes, enrich descriptions:
- Lookup product catalogs (ASIN, SKU)
- Use vendor APIs
- Manual review queue for low-confidence predictions (<90%)

### 4. Active Learning Loop
- Deploy model in shadow mode
- Flag low-confidence predictions
- Human-review corrections feed back into training
- Retrain monthly

**Out of scope for this POC**, but straightforward to implement.

## Quick Start: Using the Trained Model

The repository includes pre-trained weights and demo data. No cloud setup needed.

### Install

```bash
git clone https://github.com/yourusername/erp-classifier.git
cd erp-classifier
pip install -e .
```

### Command-Line Tool

**Interactive mode**:

```bash
python classify_erp.py

# Prompts:
#   Vendor: Amazon Services Europe
#   Description: Dell UltraSharp Monitor
#   Department: IT_OPS
#   Amount (EUR): 450

# Output:
# ╭─────────────── Prediction ───────────────╮
# │ GL_1400_IT_HARDWARE_CAPEX                │
# │ Confidence: 97.5%                        │
# ╰──────────────────────────────────────────╯
```

**One-shot prediction**:

```bash
python classify_erp.py \
  --vendor "Amazon Services Europe" \
  --description "Logitech Wireless Mouse M185" \
  --department IT_OPS \
  --amount 15

# Output:
# GL_6220_IT_PERIPHERALS_OPEX
# Confidence: 97.1%
```

**Show all probabilities**:

```bash
python classify_erp.py \
  --vendor "Amazon" \
  --description "Dell Monitor" \
  --department IT_OPS \
  --amount 450 \
  --show-all
```

Press Ctrl+C or type `quit` to exit interactive mode.

## Testing the 800 EUR Threshold

```bash
# Below threshold → OPEX
python classify_erp.py \
  --vendor "Amazon" \
  --description "Logitech Mouse" \
  --department IT_OPS \
  --amount 15
# → GL_6220_IT_PERIPHERALS_OPEX (97.1%)

# Above threshold → CAPEX
python classify_erp.py \
  --vendor "Amazon" \
  --description "Dell Monitor" \
  --department IT_OPS \
  --amount 850
# → GL_1400_IT_HARDWARE_CAPEX (97.5%)
```

## GL Accounts (8 Classes)

| GL Code | Category | Rule |
|---------|----------|------|
| GL_6220 | IT Peripherals (OPEX) | IT hardware **< 800 EUR** |
| GL_1400 | IT Hardware (CAPEX) | IT hardware **≥ 800 EUR** |
| GL_6400 | Office Consumables | Toner, paper, binders |
| GL_6410 | Facility Pantry | Coffee, snacks, beverages |
| GL_6150 | Marketing Promo | Promotional materials, swag |
| GL_6310 | Workplace Safety/Ergonomics | Ergonomic equipment, safety gear |
| GL_6250 | Tech Books/Training | Books, courses, training |
| GL_6450 | Cleaning/Maintenance | Cleaning supplies, maintenance |

## Building and Training

For technical details on training and data generation, see [BUILDING.md](BUILDING.md).

Quick links:
- **Train from demo data**: See [BUILDING.md](BUILDING.md#quick-start-train-from-pre-generated-data)
- **Generate new data**: See [BUILDING.md](BUILDING.md#generate-new-training-data)
- **LLM backend options**: See [BUILDING.md](BUILDING.md#llm-backend-options)
- **Python integration**: See [BUILDING.md](BUILDING.md#direct-python-integration)

## Data

Demo data (same as HuggingFace) is included in this repository:
- `output/03_filtered.jsonl` (942 records, human-readable)
- `output/multimodal/` (train/val/test splits, ML-ready)
- `output/evaluation-multimodal/` (confusion matrix, metrics)

**Total size**: 1.1 MB

Load from HuggingFace:
```python
from datasets import load_dataset
dataset = load_dataset("EHTB/erp-classifier-dataset")
```

## Why Not Ollama?

This is a **classification model** (predicts one of 8 GL accounts), not a generative LLM (text completion). Ollama is designed for running large language models in GGUF format for chat and text generation. The two model types are fundamentally incompatible.

Use the CLI tool (`classify_erp.py`) instead—it provides a similar terminal-based interface.

## License

MIT

## Citation

```bibtex
@misc{erp-classifier,
  title={ERP GL Account Classifier: Multimodal Tabular-Text Classification},
  author={EHTB},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/erp-classifier}}
}
```
