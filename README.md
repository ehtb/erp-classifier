# Local, domain-specific "Jev"-like ERP classifier

A classifier that decides which general ledger (GL) account an invoice line belongs to, based on what was bought, who ordered it and how much it cost. When it isn't confident, it hands the line to a person.

This is a demo showing how you can build a Jev-like model by giving it a few hard constraints. It is not a production-ready implementation, and the data is synthetic. It runs on common hardware: training takes about 6 minutes on an Apple Silicon laptop (see [docs/building.md](docs/building.md)), and predictions take milliseconds on a CPU, with no cloud GPU or hosted LLM.

## The problem

Amazon Business invoices all carry the same vendor name, so a €750 monitor and a pack of coffee pods both say "Amazon Services Europe". To pick the right account, the classifier has to read the item description, the department and the amount together.

IT hardware also has a price rule, the €800 capitalization threshold:

- Below 800 EUR -> `GL_6220` (operating expense, booked as a cost this year)
- 800 EUR and above -> `GL_1400` (capital expenditure, booked as an asset)

A €750 monitor and an €850 monitor are the same product with different accounting treatment. A model that only reads text still scores about 95% here, because titles hint at the category and price tier. But hold the description fixed and vary the price from €700 to €900, and it predicts the same account every time. This classifier is built to get that right.

## How it works

The model reads each invoice line in three parts. The text (vendor and description) goes through DistilBERT, a small pretrained language model. The department goes through a lookup table. The amount goes through its own numerical branch, built to notice sharp cut-offs like €800. Department and amount then adjust how the model reads the text, and a final layer picks one of 8 GL accounts with a confidence score. At 90% confidence or higher the line is posted automatically. Below that, a person reviews it.

The diagram, design choices and tradeoffs are in [docs/architecture.md](docs/architecture.md).

## Results

On the synthetic test set, the model picks the right account for about 99 of 100 invoice lines (98.8% ± 0.9% over 3 training seeds; the default checkpoint, seed 42, scores 97.9%). Every item close to the €800 line was correct, and as a monitor's price rises the answer switches from operating expense to capital expenditure once, near €800, and never back. With a 0.90 confidence cutoff, 95.0% of test lines are posted automatically with no errors. The other 5.0% go to a person, including €0 credit notes, items right at €800 and all three lines the default checkpoint got wrong.

Full tables, baselines and ablations are in [docs/building.md](docs/building.md#results).

## Compared with traditional ML

Here "traditional ML" means TF-IDF with logistic regression, optionally with the amount split into bins and the department added as extra columns (`scripts/baselines.py`). It runs on the same data and split.

|                          | Traditional ML                                                                                                                                         | Jev-like model                                                                                                                                                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Accuracy                 | 95.0% on text alone, 96.5% with amount bins and department                                                                                             | 98.8% (97.9% for the default checkpoint)                                                                                                                                                                                            |
| The €800 cut-off         | Text alone never switches. With amount bins it switches within 1.3% of €800                                                                            | Switches once, within 0.5% of €800, and scores 95.3% across the price sweep vs 92.6%                                                                                                                                                |
| Messy, multilingual text | Only matches whole words seen in training, so German _Bürostuhl_ and English _office chair_ are unrelated, and a typo or truncated SAP title is a miss | A multilingual pretrained language model (104 languages, including English, German and Dutch) covers synonyms across languages, typos and truncated SAP text. It is weaker on Dutch than German in a quick hand check (see the FAQ) |
| Cost                     | Cheapest to train and run on a plain CPU                                                                                                               | 135M parameters. Trains in about 6 minutes on an Apple Silicon laptop, predicts in milliseconds on a CPU, and has no per-call fees like a hosted LLM                                                                                |
| Confidence scores        | Not measured here                                                                                                                                      | Calibrated (ECE 0.035), which is what the 90% auto-post cutoff relies on                                                                                                                                                            |

On accuracy the Jev-like model is about two points ahead (98.8% vs 96.5%, roughly three test items), which is still inside the noise of a 141-item test set. If your text is clean and you only need the right account, the traditional model is the better deal. The Jev-like model earns its extra compute when the text is noisy, when a sharp price cut-off matters, and when you need trustworthy confidence scores to decide what goes to a person.

## FAQ and known limitations

This is a demo on synthetic data, and some of its results are weaker than the headline numbers suggest. The questions below are the ones a skeptical reader should ask. The technical detail is in [docs/architecture.md](docs/architecture.md#4-critical-notes-and-known-limitations).

**Why not write `if amount >= 800`?**
The rule only applies to IT hardware, and an invoice doesn't say what category it is. It has a vendor, a messy description, a department and an amount. To use the rule you would first need a classifier that finds the category, then the rule on top. This model does both in one pass. The optional two-stage price band mode (config F) is closer to that setup, and it scores best on the price sweep.

**Did the model discover the €800 threshold?**
No. It was taught it. The training set includes counterfactual copies of IT items priced on both sides of €800, generated from the rule in `src/config.py`, and the price bin edges are fit on that augmented data. Without the counterfactuals the first-crossing error goes from 0.5% to 15.4% (configs B and D in the [ablations](docs/building.md#results)), even though B scores 97.4% on plain test accuracy. That makes this rule distillation, not rule discovery. It is useful when you know the rule and want one model that applies it only to the right items. It does not show that a model can find an unknown rule from data.

**The text-only baseline gets 95%. Is the data too easy?**
Partly, yes. The synthetic generator asks for cheap-sounding items below €800 (keyboard, USB hub) and premium ones above it (4K monitor, docking station), so the wording leaks the label. Real invoices won't be that tidy. A zero-shot filter also drops generated rows where it disagrees with the target label, which removes hard cases before the split, so accuracy and calibration are probably optimistic. The fixed-text price sweep is the more honest test, because it holds the description constant and only moves the price.

**Is the test set big enough?**
Not to rank models that are close together. It has 141 items, so one item is 0.7 points, and the gap between the Jev-like model (98.8%), the best traditional baseline (96.5%) and the text-only baseline (95.0%) is two to five items, and the seed-to-seed spread alone is about one item. Only 8 test items sit near €800. The sweep adds 33 IT descriptions × 70 prices to cover that gap, and the 0.90 cutoff was picked using a handful of validation errors.

**Is the price response guaranteed to be monotone?**
No. The 16 price ramps are monotone, but the layers after them are not constrained, so monotone behavior is a strong bias and not a proof. On the 33 IT test items the largest drop in probability as price rises is 0.03% for the default checkpoint and 0.2% at worst across the three seeds (config C, with the sine/cosine encoder, drops up to 20%). The "0% non-monotone" figure ignores drops under 5%.

## Quick start

The rest of this page is hands-on setup for engineers. The repository includes the pre-generated dataset splits in `output/multimodal/`. Train a checkpoint locally (about 6 minutes on Apple Silicon; see [docs/building.md](docs/building.md)), then run the CLI or browser demo.

### Install

```bash
git clone https://github.com/EHTB/erp-classifier.git
cd erp-classifier
pip install -e .
```

### Command-line tool

Sweep a single item across the 800 EUR boundary:

```bash
python classify_erp.py --sweep 650:950:50
```

One-shot prediction with confidence routing:

```bash
python classify_erp.py \
  --vendor "Amazon Services Europe" \
  --description "Dell UltraSharp Monitor" \
  --department IT_OPS \
  --amount 850

# Output:
# ╭─────────────────────── Prediction ───────────────────────╮
# │ GL_1400_IT_HARDWARE_CAPEX                                │
# │ Confidence: 97.4%                                        │
# │ Routing:    AUTO-POST (confidence >= 90% configured      │
# │             cutoff)                                      │
# ╰──────────────────────────────────────────────────────────╯
```

Show all class probabilities and FiLM modulation stats:

```bash
python classify_erp.py \
  --vendor "Amazon Services Europe" \
  --description "Dell UltraSharp Monitor" \
  --department IT_OPS \
  --amount 750 \
  --show-all
```

Interactive prompt mode:

```bash
python classify_erp.py
```

### Browser demo (Gradio)

Launch the interactive web UI with preset invoice lines, an amount slider, FiLM gate diagnostics, and a live counterfactual sweep plot:

```bash
pip install -e ".[demo]"
python demo/app.py
```

## Testing the 800 EUR threshold

Hold the vendor, description, and department fixed while changing the price across 800 EUR:

```bash
# Below threshold (750 EUR) -> OPEX
python classify_erp.py \
  --vendor "Amazon Services Europe" \
  --description "Dell UltraSharp Monitor" \
  --department IT_OPS \
  --amount 750
# -> GL_6220_IT_PERIPHERALS_OPEX (97.6% confidence, AUTO-POST)

# Above threshold (850 EUR) -> CAPEX
python classify_erp.py \
  --vendor "Amazon Services Europe" \
  --description "Dell UltraSharp Monitor" \
  --department IT_OPS \
  --amount 850
# -> GL_1400_IT_HARDWARE_CAPEX (97.4% confidence, AUTO-POST)
```

With text and department unchanged, the prediction switches from `GL_6220` to `GL_1400` within about 1% of 800 EUR. Between 800 and 801 EUR confidence drops to about 51–54%, below the 0.90 cutoff, so those invoices go to manual review.

## Building and training

For architecture details, training instructions, synthetic data generation, and ablation reproduction, see [docs/architecture.md](docs/architecture.md), [docs/building.md](docs/building.md), and [CHANGELOG.md](CHANGELOG.md).

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
