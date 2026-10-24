# ERP GL Account Classifier

A multimodal ML classifier that predicts GL accounts for ERP transactions by learning from product descriptions, transaction amounts, and department codes together.

## The Problem

Amazon Business invoices all show the same vendor name. A €750 monitor and a pack of coffee pods both say "Amazon Services Europe" — the model has to look at department, amount, and description together to classify them correctly.

IT hardware has a special challenge: it splits across two GL accounts based on price alone:
- **< 800 EUR** → GL_6220 (operating expense)
- **≥ 800 EUR** → GL_1400 (capital expenditure)

A €750 monitor and an €850 monitor are the same item, different accounting treatment. A text-only classifier scores about 95% on this dataset, because descriptions already hint at the category and price tier. But it never sees the amount. Keep the description the same and it gives the same GL account at €700 and at €900.

## The Solution: Multimodal Architecture (v0.4.0)

Instead of forcing text models to parse numbers, we treat ERP transactions as **hybrid tabular-text data** with a dedicated numerical branch:

```
┌─────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
│  Text Features          │      │ Categorical Features     │      │ Continuous Features      │
│  vendor + description   │      │ department               │      │ [z_score, log1p(amt)]    │
└───────────┬─────────────┘      └──────────┬───────────────┘      └───────────┬──────────────┘
            │                               │                                   │
            ▼                               ▼                                   ▼
┌─────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
│ DistilBERT Encoder      │      │ Embedding Layer          │      │ Piecewise-Linear Encoding│
│ (pretrained)            │      │ dept → 32-dim            │      │ data-learned bin edges   │
│                         │      │ (with 15% UNK dropout)   │      │ 2-dim → 2 + 16 ramps     │
└───────────┬─────────────┘      └──────────┬───────────────┘      └───────────┬──────────────┘
            │                               │                                   │
            │                               │                                   ▼
            │                               │                       ┌──────────────────────────┐
            │                               │                       │ Numerical Projection     │
            │                               │                       │ 2-layer MLP: 18→64       │
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

What's new in v0.4.0:

1. **Piecewise-linear amount encoding.** The amount goes through ramps between bin edges taken from the training data, so a higher amount can only push the score in one direction. It is the default; `--num-encoding periodic` in `train.py` brings back the older sin/cos encoder.
2. **Counterfactual training copies.** For each IT record, training adds 4 copies with the amount moved near the 800 EUR line and the label recomputed from the rules (`src/boundaries.py`). An original and its copies share one unit of weight (1/5 each), so a product family still counts once. It is the default; `--counterfactuals-per-record 0` in `build_dataset.py` turns it off.
3. **Price scrubbing and grouped splits.** We strip prices written into descriptions at build time and again at inference (`src/text.py`). Near-duplicate product titles always land in the same split, so the test set can't contain a twin of a training item. This part is always on.
4. **FiLM gating, department dropout (15%), label smoothing (0.05).** These carry over from earlier versions.

The price band mode described under Results is a separate option and is off by default.

## Results

### Current Performance (v0.4.0)

Results come from 3 training seeds on one fixed split: 659 original training records plus 644 counterfactual copies, 141 validation records and 141 test records (87 product groups in the test set).

- **Test accuracy:** 96.5% ± 0.6% (seed 42: 97.2%; macro F1 0.965; ECE 0.046)
- **Items within ±12.5% of 800 EUR:** 8 of 8 correct for every seed
- **Amount sweep** (33 IT test items, 70 amounts each, text held fixed): the prediction flips from OPEX to CAPEX exactly once in 100% of cases and never goes back (0% non-monotone). It is right everywhere outside a ±2% band around the boundary. The first crossing lands 0.60% away from the true boundary (median).

| Model | Test Accuracy | Near-Boundary | Single Flip | Tolerance Step (±2%) | Crossing Error |
|-------|---------------|---------------|-------------|----------------------|----------------|
| Majority class | 12.8% | 4/8 | — | — | — |
| Amount + department (GBDT) | 65.2% | 8/8 | 100% | 100% | 0.6% |
| TF-IDF + LR (text only) | 95.0% | 6/8 | 0% | 0% | n/a |
| TF-IDF + binned amount + dept (LR) | 96.5% | 8/8 | 100% | 100% | 1.3% |
| **Multimodal PLE + CF (default)** | **96.5%** | **8/8** | **100%** | **100%** | **0.6%** |

The counterfactual copies help the simple binned baseline as well as the neural model. The neural model finds the boundary about twice as precisely (0.6% vs 1.3% error) and is a bit more accurate across the sweep (94.8% vs 92.6%), but the two tie on test accuracy. The simple baseline is much cheaper to run, so the neural model is worth it only if you need the smooth, one-way price behaviour. Full ablation tables are in [`output/evaluation-multimodal/ablation_summary.md`](output/evaluation-multimodal/ablation_summary.md).

### Confidence Routing

The cutoff is the lowest confidence at which validation accuracy reaches 99%. That gives **0.90** (seed 42):
- **Auto-posted:** 133 of 141 test items (94.3%), all correct
- **Sent to manual review:** 8 of 141 (5.7%)

### Optional: price band mode

Build the dataset with `--price-band` and a first step labels each amount as below or above the 800 EUR line, using the accounting rules. The classifier then gets that label as a token at the start of its text. The OPEX to CAPEX switch lands right on 800 instead of about 0.6% off. Overall accuracy stays the same (95.7% vs 96.5%, within noise), since the model already got every real item near 800 right. The mode is off by default, because it hands the model the rule instead of letting it learn it.

```bash
python scripts/build_dataset.py --input output/03_filtered.jsonl \
  --output-dir output/multimodal-band --price-band
```

Checkpoints trained on this data carry a `price_band.json`, and `classify_erp.py` applies it on its own. Details are in [BUILDING.md](BUILDING.md).

## Quick Start: Using the Trained Model

The repository includes the pre-generated dataset splits in `output/multimodal/`. Train a checkpoint locally first (see [BUILDING.md](BUILDING.md)), then run the CLI:

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
# │ Confidence: 97.7%                        │
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

Counterfactual test: same item, different price → different GL account:

```bash
# Below threshold (750 EUR) → OPEX
python classify_erp.py \
  --vendor "Amazon Services Europe" \
  --description "Dell UltraSharp Monitor" \
  --department IT_OPS \
  --amount 750
# → GL_6220_IT_PERIPHERALS_OPEX (97.7% confidence)

# Above threshold (850 EUR) → CAPEX
python classify_erp.py \
  --vendor "Amazon Services Europe" \
  --description "Dell UltraSharp Monitor" \
  --department IT_OPS \
  --amount 850
# → GL_1400_IT_HARDWARE_CAPEX (97.7% confidence)
```

With the text and department unchanged, the prediction moves from `GL_6220` to `GL_1400` within about 1% of the 800 EUR line. Right on the line (800–801 EUR) the model is unsure, at 64–68% confidence, so those cases fall under the 0.90 cutoff and go to manual review.

## Building and Training

For technical details on training, ablations, and data generation, see [BUILDING.md](BUILDING.md) and [CHANGELOG.md](CHANGELOG.md).

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
