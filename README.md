# ERP GL Account Classifier

A multimodal ML classifier that predicts GL accounts for ERP transactions by learning from product descriptions, transaction amounts, and department codes together.

**Demo**: [🤗 HuggingFace Dataset](https://huggingface.co/datasets/EHTB/erp-classifier-dataset) | [🤖 Trained Model](https://huggingface.co/EHTB/erp-classifier)

## The Problem

Amazon Business invoices all show the same vendor name. A €750 monitor and a pack of coffee pods both say "Amazon Services Europe" — the model has to look at department, amount, and description together to classify them correctly.

IT hardware has a special challenge: it splits across two GL accounts based on price alone:
- **< 800 EUR** → GL_6220 (operating expense)
- **≥ 800 EUR** → GL_1400 (capital expenditure)

A €750 monitor and an €850 monitor are the same item, different accounting treatment. Text-only classifiers fail here because they see "[NET_AMT] 799.00" as text tokens, not numbers—they can't learn mathematical thresholds.

## The Solution: Multimodal Architecture (v0.3.0)

Instead of forcing text models to parse numbers, we treat ERP transactions as **hybrid tabular-text data** with advanced architectural components:

```
┌─────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
│  Text Features          │      │ Categorical Features     │      │ Continuous Features      │
│  vendor + description   │      │ department               │      │ [z_score, log1p(amt)]    │
└───────────┬─────────────┘      └──────────┬───────────────┘      └───────────┬──────────────┘
            │                               │                                   │
            ▼                               ▼                                   ▼
┌─────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
│ DistilBERT Encoder      │      │ Embedding Layer          │      │ Periodic Encoding        │
│ (pretrained)            │      │ dept → 32-dim            │      │ learnable frequencies    │
│                         │      │ (with 15% UNK dropout)   │      │ 2-dim → 66-dim           │
└───────────┬─────────────┘      └──────────┬───────────────┘      └───────────┬──────────────┘
            │                               │                                   │
            │                               │                                   ▼
            │                               │                       ┌──────────────────────────┐
            │                               │                       │ Numerical Projection     │
            │                               │                       │ 2-layer MLP: 66→64       │
            │                               │                       └───────────┬──────────────┘
            │                               │                                   │
            │                               └────────────┬──────────────────────┘
            │                                            │
            │                                            ▼
            │                               ┌──────────────────────────┐
            │                               │   FiLM Gating            │
            │                               │   γ, β = MLP(num+dept)   │
            │                               └──────────┬───────────────┘
            │                                          │
            └──────────────┬───────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────────────────┐
            │  Modulated Text: cls * (1+tanh(γ)) + β   │
            └──────────────┬───────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────────────────┐
            │   Fusion Classifier                      │
            │   [text, num, dept] → MLP → 8 GL codes   │
            │   864-dim → 256 → 128 → 8                │
            └──────────────────────────────────────────┘
```

Architectural features:

1. Periodic Encoding: Learns high-frequency resolution around the 800 EUR threshold via learnable sin/cos transformations
2. FiLM Gating: Tabular features (amount + department) modulate text embeddings for department-aware interpretation  
3. Modality Dropout: 15% dept → UNK during training improves generalization
4. Label Smoothing: 0.05 entropy regularization prevents overconfidence

The model learns the "< 800 vs ≥ 800" boundary through periodic encoding + backpropagation, not text parsing or hardcoded rules.

## Results

### Current Performance (v0.3.0)

Performance (659 training samples, multimodal architecture with periodic encoding and FiLM gating):

- Validation Accuracy: 97.87%
- Test Accuracy: 97.18%

At 0.90 confidence threshold:
- Auto-post: 133/142 test samples (93.7%) at 100% accuracy
- Manual review: 9/142 samples (6.3%) - primarily €0 edge cases and low-confidence predictions
- Zero incorrect auto-posts

| Confidence Threshold | Accuracy | Coverage | Status |
|---------------------|----------|----------|--------|
| 0.50 | 97.18% | 100% | ⚠ Review needed |
| 0.70 | 100% | 96.48% | ✓ Aggressive automation |
| 0.80 | 100% | 95.77% | ✓ Balanced |
| 0.90 (recommended) | 100% | 93.66% | ✓ Production-ready |
| 0.95 | 100% | 92.25% | ✓ Maximum confidence |

## Quick Start: Using the Trained Model

The repository includes pre-trained weights and demo data. No cloud setup needed.

### Install

```bash
git clone https://github.com/EHTB/erp-classifier.git
cd erp-classifier
pip install -e .
```

### Command-Line Tool

Interactive mode:

```bash
python classify_erp.py

# Prompts:
#   Vendor: Amazon Services Europe
#   Description: Dell UltraSharp Monitor
#   Department: IT_OPS
#   Amount (EUR): 850

# Output:
# ╭─────────────── Prediction ───────────────╮
# │ GL_1400_IT_HARDWARE_CAPEX                │
# │ Confidence: 98.3%                        │
# ╰──────────────────────────────────────────╯
```

One-shot prediction:

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

Show all probabilities:

```bash
python classify_erp.py \
  --vendor "Amazon Services Europe" \
  --description "Dell UltraSharp Monitor" \
  --department IT_OPS \
  --amount 750 \
  --show-all
```

Press Ctrl+C or type `quit` to exit interactive mode.

## Testing the 800 EUR Threshold

Counterfactual Test: Same item, different price → different GL account

```bash
# Below threshold (750 EUR) → OPEX
python classify_erp.py \
  --vendor "Amazon Services Europe" \
  --description "Dell UltraSharp Monitor" \
  --department IT_OPS \
  --amount 750
# → GL_6220_IT_PERIPHERALS_OPEX (60.5% confidence)

# Above threshold (850 EUR) → CAPEX
python classify_erp.py \
  --vendor "Amazon Services Europe" \
  --description "Dell UltraSharp Monitor" \
  --department IT_OPS \
  --amount 850
# → GL_1400_IT_HARDWARE_CAPEX (98.3% confidence)
```

Identical text ("Dell UltraSharp Monitor"), different amounts → model correctly applies the 800 EUR capitalization rule via learned periodic encoding, not text pattern matching.

## Building and Training

For technical details on training and data generation, see [BUILDING.md](BUILDING.md).

## License

MIT

## Citation

```bibtex
@misc{erp-classifier,
  title={ERP GL Account Classifier: Multimodal Tabular-Text Classification},
  author={EHTB},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/EHTB/erp-classifier}}
}
```
