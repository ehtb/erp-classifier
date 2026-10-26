# Local, domain-specific "Jev"-like ERP classifier

A classifier that decides which general ledger (GL) account an invoice line belongs to, based on what was bought, who ordered it and how much it cost. When it isn't confident, it hands the line to a person.

This is a demo showing how you can build a Jev-like model by giving it a few hard constraints. It is not a production-ready implementation, and the data is synthetic. It runs on common hardware: training takes about 4 minutes on an Apple Silicon laptop (see [docs/building.md](docs/building.md)), and predictions take milliseconds on a CPU, with no cloud GPU or hosted LLM.

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

On the synthetic test set, the model picks the right account for about 96 of 100 invoice lines (96.5% ± 0.6% over 3 training seeds). Every item close to the €800 line was correct, and as a monitor's price rises the answer switches from operating expense to capital expenditure once, near €800, and never back. With a 0.90 confidence cutoff, 94.3% of test lines are posted automatically with no errors. The other 5.7% go to a person, including €0 credit notes and items right at €800.

Full tables, baselines and ablations are in [docs/building.md](docs/building.md#results).

## Quick start

The rest of this page is hands-on setup for engineers. The repository includes the pre-generated dataset splits in `output/multimodal/`. Train a checkpoint locally (about 4 minutes on Apple Silicon; see [docs/building.md](docs/building.md)), then run the CLI or browser demo.

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
# │ Confidence: 97.7%                                        │
# │ Routing:    AUTO-POST (confidence >= 90% validation      │
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
# -> GL_6220_IT_PERIPHERALS_OPEX (97.7% confidence, AUTO-POST)

# Above threshold (850 EUR) -> CAPEX
python classify_erp.py \
  --vendor "Amazon Services Europe" \
  --description "Dell UltraSharp Monitor" \
  --department IT_OPS \
  --amount 850
# -> GL_1400_IT_HARDWARE_CAPEX (97.7% confidence, AUTO-POST)
```

With text and department unchanged, the prediction switches from `GL_6220` to `GL_1400` within about 1% of 800 EUR. Between 800 and 801 EUR confidence drops to 64–68%, below the 0.90 cutoff, so those invoices go to manual review.

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
