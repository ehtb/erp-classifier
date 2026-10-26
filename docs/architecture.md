# ERP classifier architecture

This document walks through how the General Ledger (GL) account classifier works, how an invoice moves through the code (with file and line references), and what each design choice costs.

> This is a demo showing how you can build a Jev-like model by giving it a few hard constraints. It is not a production-ready implementation. The data is synthetic.

---

## 1. Why three branches

Amazon Business invoices all carry nearly the same vendor name (`Amazon Services Europe`, `AMZN Mktp DE`, `Amazon.nl`; see [`src/config.py:150-158`](../src/config.py#L150-L158)). A €18.50 bag of coffee beans and a €1,190 laptop both say "Amazon", so the vendor alone settles nothing.

Picking one of the 8 GL accounts ([`src/config.py:21-31`](../src/config.py#L21-L31)) takes three things:
1. What was bought (vendor and item description)
2. Who ordered it (department code, such as `IT_OPS`)
3. How much it cost (net amount in EUR)

IT equipment also has a hard €800 capitalization threshold ([`src/config.py:97-107`](../src/config.py#L97-L107)):
- Below €800 $\rightarrow$ `GL_6220_IT_PERIPHERALS_OPEX` (operating expense)
- €800 and above $\rightarrow$ `GL_1400_IT_HARDWARE_CAPEX` (capital asset)

### Why not feed the price as text?
A text model sees `"750"` and `"850"` as unrelated tokens, with no sense that `799.99` and `800.00` are neighbors. It ends up guessing the price from words like *"UltraSharp"* and predicts the same account at €700 and at €900.

So the model splits each invoice into three branches (text, department, euro amount) and combines them later:

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

---

## 2. Walkthrough

### Branch 1: Text Features $\rightarrow$ DistilBERT Encoder

```
┌─────────────────────────┐
│  Text Features          │
│  vendor + description   │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ DistilBERT Encoder      │
│ (pretrained)            │
└─────────────────────────┘
```

#### How it works
The left branch reads the text. Invoice lines are messy: English, German or Dutch ([`src/config.py:44-50`](../src/config.py#L44-L50)), cryptic part numbers (`ASIN_SKU_HEAVY`), or cut off at 40 characters by older SAP systems (`SAP_SGTXT_ABBREVIATED`, [`src/config.py:53-59`](../src/config.py#L53-L59) and [`src/corruptor.py`](../src/corruptor.py)).

First, `scrub_prices()` ([`src/text.py:45-52`](../src/text.py#L45-L52)) strips euro amounts written in the description (such as `"... for 678 EUR"`), both when building the dataset ([`src/multimodal_dataset.py:287`](../src/multimodal_dataset.py#L287)) and at prediction ([`classify_erp.py:193`](../classify_erp.py#L193)). Otherwise the model would learn price words instead of using the numerical branch.

Vendor and description are then combined into one tagged string ([`classify_erp.py:194`](../classify_erp.py#L194), [`src/multimodal_dataset.py:177`](../src/multimodal_dataset.py#L177)) and fed into DistilBERT ([`src/model.py:162-171`](../src/model.py#L162-L171)), a small pretrained language model. It returns 768 numbers (the `[CLS]` token, extracted at [`src/model.py:249-250`](../src/model.py#L249-L250)).

#### Examples
- Clean IT hardware preset ([`demo/app.py:30`](../demo/app.py#L30)):
  - Input fields: `vendor="Amazon Services Europe"`, `description="Dell UltraSharp U2723DE 27 Monitor"`
  - Formatted string sent to DistilBERT ([`classify_erp.py:194`](../classify_erp.py#L194)):
    ```text
    [VENDOR] Amazon Services Europe [DESC] Dell UltraSharp U2723DE 27 Monitor
    ```
- Abbreviated, SKU-heavy laptop line ([`demo/app.py:33`](../demo/app.py#L33)):
  - Input: `vendor="AMZN Mktp DE"`, `description="DELL LAT 5540 CTO I7/16/512GB/15.6FHD"`
- Truncated SAP text in training data ([`output/multimodal/train_multimodal.jsonl:5`](../output/multimodal/train_multimodal.jsonl#L5)):
  - `[VENDOR] Amazon.nl [DESC] Premium Quality Office Paper - A4 Siz...`

#### Tradeoffs
- Why not a hosted LLM on every invoice? It costs money per call and returns uncalibrated confidence. DistilBERT (~66.8M parameters, [`train.py:277-280`](../train.py#L277-L280)) runs locally in milliseconds.
- Why not TF-IDF? It can't connect synonyms or languages (German *Bürostuhl* vs. English *office chair*).
- The catch: a high learning rate would wreck what DistilBERT already knows. Training uses `3e-5` for it (`--learning-rate`, [`src/config.py:237`](../src/config.py#L237)) and `5e-4`, about 16x higher, for the new price and department layers (`--tabular-lr`, [`src/config.py:238`](../src/config.py#L238), [`train.py:314-355`](../train.py#L314-L355)).

---

### Branch 2: Categorical Features $\rightarrow$ Embedding Layer (with 15% UNK dropout)

```
┌──────────────────────────┐
│ Categorical Features     │
│ department               │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ Embedding Layer          │
│ dept → 32-dim            │
│ (with 15% UNK dropout)   │
└──────────────────────────┘
```

#### How it works
The middle branch encodes who ordered the item. Each department (`ENG`, `FACILITIES`, `HR`, `IT_OPS`, `MARKETING`) gets an ID from `1` to `5`, with `0` for `UNK` (unknown) ([`src/multimodal_dataset.py:46-69`](../src/multimodal_dataset.py#L46-L69), stored in [`output/multimodal/dept_vocab.json:1-18`](../output/multimodal/dept_vocab.json#L1-L18)). An embedding layer ([`src/model.py:194-197`](../src/model.py#L194-L197)) turns the ID into a 32-number vector of purchasing habits (see `CATEGORY_DEPT_AFFINITY` in [`src/config.py:162-170`](../src/config.py#L162-L170)).

##### Why the 15% UNK dropout?
In training, 15% of invoices get their department replaced with `0` ([`src/model.py:205`](../src/model.py#L205), [`src/model.py:256-262`](../src/model.py#L256-L262)). This teaches the model to handle a missing department. It also stops it from shortcutting (*"`MARKETING` means `GL_6150_MARKETING_PROMO`"*) when `MARKETING` also orders office consumables in `GL_6400` ([`src/config.py:164`](../src/config.py#L164)).

#### Examples
- Known department ([`demo/app.py:34`](../demo/app.py#L34)):
  - `"Lavazza Qualita Rossa Koffiebonen 1kg"` ordered by `FACILITIES` $\rightarrow$ `dept_vocab.get("FACILITIES", 0)` returns ID `2` ([`classify_erp.py:210`](../classify_erp.py#L210), [`output/multimodal/dept_vocab.json:4`](../output/multimodal/dept_vocab.json#L4)), which reinforces `GL_6410_FACILITY_PANTRY`.
- Missing or unknown department ([`demo/app.py:35`](../demo/app.py#L35)):
  - `["Amazon EU S.a.r.l.", "Credit note adjustment", "UNK", 0.0]` $\rightarrow$ maps to ID `0` (`UNK`). Since 15% of training examples used `0` ([`src/model.py:259-260`](../src/model.py#L259-L260)), that vector is a trained neutral signal, not noise.

#### Tradeoffs
- Pro: tiny (`6 × 32 = 192` learnable numbers), and missing metadata doesn't break anything.
- Con: fixed vocabulary. A new department (say `LEGAL`) falls back to `UNK` at [`classify_erp.py:210`](../classify_erp.py#L210) instead of crashing, but the model knows nothing about it until retraining.

---

### Branch 3: Continuous Features $\rightarrow$ Piecewise-Linear Encoding $\rightarrow$ Numerical Projection

```
┌──────────────────────────┐
│ Continuous Features      │
│ [z_score, log1p(amt)]    │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│ Piecewise-Linear Encoding│
│ data-learned bin edges   │
│ 2-dim → 2 + 16 ramps     │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│ Numerical Projection     │
│ 2-layer MLP: 18→64       │
└──────────────────────────┘
```

#### How it works

##### Step 3a: Why two versions of the same euro amount (`[z_score, log1p(amt)]`)?
Amounts run from €5 pens to €2,500 workstations ([`PRICE_RANGES` in `src/config.py:174-182`](../src/config.py#L174-L182)). Each amount becomes two numbers ([`src/multimodal_dataset.py:181-187`](../src/multimodal_dataset.py#L181-L187), [`classify_erp.py:206-208`](../classify_erp.py#L206-L208)):
1. `z_score`: `(amount - 339.09) / 491.45`, using the training mean and standard deviation from [`output/multimodal/scaler.json:1-10`](../output/multimodal/scaler.json#L1-L10). Keeps euro differences proportional.
2. `log1p(amt)`: `(log(1 + amount) - 5.075) / 1.205`. Makes €10 vs. €50 as visible as €750 vs. €850. ([`classify_erp.py:205-206`](../classify_erp.py#L205-L206) clamps negative credit-note amounts to `0.0` first.)

##### Step 3b: What is Piecewise-Linear Encoding (PLE) (`2-dim -> 2 + 16 ramps`)?
A linear layer scales its output with the input, so it can't easily learn *"do nothing from €50 to €750, then flip between €780 and €820"*. Piecewise-linear encoding (`PiecewiseLinearEncoding` in [`src/model.py:67-93`](../src/model.py#L67-L93)) fixes this with a 16-bar thermometer:
- Training prices are split into 16 quantile bins (`fit_bin_edges()` in [`src/multimodal_dataset.py:512-547`](../src/multimodal_dataset.py#L512-L547), saved in [`output/multimodal/numeric_bins.json:1-24`](../output/multimodal/numeric_bins.json#L1-L24)).
- Each bin is a ramp ([`src/model.py:90-91`](../src/model.py#L90-L91)): `0.0` below the bin, `1.0` above it, the exact fraction inside it.
- The 16 bars are appended to the 2 original numbers, giving `2 + 16 = 18` inputs ([`src/model.py:87`](../src/model.py#L87), [`src/model.py:183`](../src/model.py#L183)).
- Counterfactual price copies around €800 ([`src/multimodal_dataset.py:320-327`](../src/multimodal_dataset.py#L320-L327), [`src/multimodal_dataset.py:469-498`](../src/multimodal_dataset.py#L469-L498)) cluster training prices there, so the bin edges ([`output/multimodal/numeric_bins.json:13-21`](../output/multimodal/numeric_bins.json#L13-L21)) pack tightly around the boundary.

##### Step 3c: Numerical Projection (`2-layer MLP: 18->64`)
A 2-layer MLP (`self.num_proj`, [`src/model.py:185-192`](../src/model.py#L185-L192)) expands the 18 numbers to 64. [`train.py:342-348`](../train.py#L342-L348) disables weight decay on its biases so it can't dull the €800 threshold.

#### Example: €750 vs. €850 monitor (`demo/app.py:30-31`)
The Dell UltraSharp monitor at €750 and €850, using [`output/multimodal/scaler.json`](../output/multimodal/scaler.json) and [`output/multimodal/numeric_bins.json`](../output/multimodal/numeric_bins.json):
- €750.00 ([`demo/app.py:30`](../demo/app.py#L30)): log-amount `(ln(751) - 5.0753) / 1.2047 = +1.283`, inside bin 10 (`1.2715` to `1.3417`, [`numeric_bins.json:16-17`](../output/multimodal/numeric_bins.json#L16-L17)). Bins 1–9 read `1.0`, bin 10 reads `0.17`, bins 11–16 read `0.0`.
- €850.00 ([`demo/app.py:31`](../demo/app.py#L31)): log-amount `(ln(851) - 5.0753) / 1.2047 = +1.386`, inside bin 11 (`1.3417` to `1.3928`, [`numeric_bins.json:17-18`](../output/multimodal/numeric_bins.json#L17-L18)). Bins 1–10 read `1.0`, bin 11 reads `0.87`, bins 12–16 read `0.0`.

Bins 10 and 11 flip from `0` to `1` across €800, so the network only needs a positive weight on them to switch from OPEX (`GL_6220`) to CAPEX (`GL_1400`).

#### Tradeoffs
- Why not sine/cosine (`PeriodicNumericalEncoding`, [`src/model.py:23-64`](../src/model.py#L23-L64))? They repeat, so the model could flip to CAPEX at €800 and *back* to OPEX at €2,000. PLE bars only go up, so a higher price pushes the score one way. Across 33 IT test items: 100% single flip, 0% non-monotone ([`building.md`](building.md#results)).
- Why not hardcode `if amount >= 800`? The rule only applies to IT hardware, and the model doesn't know in advance which invoices those are. Learning it from data fires the switch only when text and department point to IT hardware. The cost: the learned boundary lands within ~0.6% of €800 (roughly €795–€805), not exactly on it, unless you use the optional `--price-band` mode ([`src/text.py:70-107`](../src/text.py#L70-L107)).

---

### Stage 4: FiLM gating and modulated text

```
                        ┌──────────────────────────┐
                        │   FiLM Gating            │
                        │   γ, β = MLP(num+dept)   │
                        └──────────┬───────────────┘
                                   │
     ┌─────────────────────────────┘
     ▼
┌──────────────────────────────────────────┐
│  Modulated Text: cls * (1+tanh(γ)) + β   │
└──────────────────────────────────────────┘
```

#### How it works
We now have 768 text numbers (`cls_raw`, [`src/model.py:250`](../src/model.py#L250)), 64 price numbers (`num_vec`, [`src/model.py:254`](../src/model.py#L254)) and 32 department numbers (`dept_vec`, [`src/model.py:262`](../src/model.py#L262)). Concatenated, the text would outnumber the rest 8 to 1, and an expensive-sounding title (*"UltraSharp 27 Monitor"*) would drown out the price.

FiLM (Feature-wise Linear Modulation) lets price and department rescale and shift each text dimension ([`src/model.py:199-202`](../src/model.py#L199-L202), [`src/model.py:264-272`](../src/model.py#L264-L272)):
1. The 64 price and 32 department numbers become a 96-number vector (`tabular_vec`, [`src/model.py:265`](../src/model.py#L265)).
2. A linear layer (`self.film_gate`, [`src/model.py:202`](../src/model.py#L202), [`src/model.py:266-267`](../src/model.py#L266-L267)) produces two sets of 768 values:
   - $\gamma$ (scale): passed through `1 + tanh(γ)` ([`src/model.py:268`](../src/model.py#L268)), a multiplier from `0.0` to `2.0`. It starts near `1.0x` (untouched) because `1 + tanh(0) = 1`.
   - $\beta$ (shift): added to each text dimension ([`src/model.py:271`](../src/model.py#L271)).
3. The result is `cls_mod = cls_raw * (1 + tanh(γ)) + β` ([`src/model.py:271`](../src/model.py#L271)).

#### Examples
FiLM shows up in the Gradio demo ([`demo/app.py:158-164`](../demo/app.py#L158-L164)) and in `python classify_erp.py --show-all` ([`classify_erp.py:238-241`](../classify_erp.py#L238-L241), [`classify_erp.py:294-300`](../classify_erp.py#L294-L300)), both via `forward_with_diagnostics()` ([`src/model.py:343-371`](../src/model.py#L343-L371)):
- `film_scale_mean` ([`src/model.py:366`](../src/model.py#L366)): average `1 + tanh(γ)` across the 768 dimensions.
- `film_shift_norm` ([`src/model.py:367`](../src/model.py#L367)): size of the `β` shift.
- `relative_text_modulation` ([`src/model.py:362`](../src/model.py#L362), [`src/model.py:368`](../src/model.py#L368)): `||cls_mod - cls_raw|| / ||cls_raw||`, how much price and department changed the text vector.

#### Tradeoffs
- Pro: price and department interact with the text multiplicatively before the classifier, and the diagnostics make that visible.
- Con: adds `96 × 1,536 = 147,456` weights (`self.film_gate`, [`src/model.py:202`](../src/model.py#L202)), and everything after FiLM re-runs when the price changes (DistilBERT only depends on the text).

---

### Stage 5: Fusion Classifier (`864-dim -> 256 -> 128 -> 8`)

```
┌──────────────────────────────────────────┐
│   Fusion Classifier                      │
│   [text, num, dept] → MLP → 8 GL codes   │
│   864-dim → 256 → 128 → 8                │
└──────────────────────────────────────────┘
```

#### How it works
The fusion classifier ([`src/model.py:207-219`](../src/model.py#L207-L219), [`src/model.py:273-275`](../src/model.py#L273-L275)) makes the final decision:
1. It concatenates `cls_mod` (768), `num_vec` (64) and `dept_vec` (32) into one 864-number vector (`fused`, [`src/model.py:274`](../src/model.py#L274)). Price and department go in a second time, unmixed, so the classifier can tell whether a signal came from the wording or the amount.
2. A 3-layer MLP (`864 → 256 → 128 → 8`, [`src/model.py:209-219`](../src/model.py#L209-L219)) outputs 8 scores (logits), one per GL account ([`src/config.py:21-31`](../src/config.py#L21-L31)).
3. Confidence and routing:
   - Training uses 5% label smoothing (`label_smoothing=0.05`, [`src/model.py:318`](../src/model.py#L318), [`src/model.py:328`](../src/model.py#L328)) so the model doesn't become 100% confident on ambiguous items.
   - Each original IT item and its 4 counterfactual copies get `sample_weight = 0.2` ([`src/multimodal_dataset.py:320-327`](../src/multimodal_dataset.py#L320-L327), [`src/model.py:313-322`](../src/model.py#L313-L322)), so the extra copies teach the boundary without making IT hardware look 5× more common.
   - At prediction ([`classify_erp.py:219-222`](../classify_erp.py#L219-L222)), a top probability of 90% or higher (`DEFAULT_CONFIDENCE_CUTOFF = 0.90`, [`classify_erp.py:49`](../classify_erp.py#L49)) is marked `AUTO_POST`. Anything lower goes to `MANUAL_REVIEW`.

#### Examples: auto-post vs. manual review
Three Dell monitor presets from [`demo/app.py:30-32`](../demo/app.py#L30-L32) (and the CLI sweep in [`classify_erp.py:303-388`](../classify_erp.py#L303-L388)):
1. €750.00 ([`demo/app.py:30`](../demo/app.py#L30)): `GL_6220_IT_PERIPHERALS_OPEX`, 97.7% confidence $\rightarrow$ `AUTO_POST`.
2. €850.00 ([`demo/app.py:31`](../demo/app.py#L31)): `GL_1400_IT_HARDWARE_CAPEX`, 97.7% confidence $\rightarrow$ `AUTO_POST`.
3. €800.00 ([`demo/app.py:32`](../demo/app.py#L32)) or a €0 credit note ([`demo/app.py:35`](../demo/app.py#L35)): at €800.00 the probability splits between OPEX and CAPEX (~64%–68% top-1). Since `68% < 90%`, [`classify_erp.py:222`](../classify_erp.py#L222) and [`demo/app.py:147-150`](../demo/app.py#L147-L150) send it to `MANUAL_REVIEW`.

On the test set, the 90% cutoff auto-posts 94.3% of invoices (133/141) with 100% precision and sends the other 8 borderline or unusual cases to review ([`building.md`](building.md#confidence-routing)).

---

## 3. Tradeoffs at a glance

| Design Choice | Where in Code | What It Buys You (Pros) | What It Costs You (Cons) |
| :--- | :--- | :--- | :--- |
| **Price Scrubbing** (`scrub_prices`) | [`src/text.py:45-52`](../src/text.py#L45-L52)<br>[`classify_erp.py:193`](../classify_erp.py#L193) | Prevents the text encoder from memorizing numbers written inside descriptions; forces price logic through the numerical branch. | Requires regex preprocessing on every training and inference description. |
| **DistilBERT Text Branch** (`768-dim`) | [`src/model.py:162-171`](../src/model.py#L162-L171)<br>[`src/model.py:249-250`](../src/model.py#L249-L250) | Understands EN/DE/NL descriptions, typos, and 40-char SAP truncations locally in milliseconds. | ~66.8M parameters; requires a smaller learning rate (`3e-5` vs `5e-4`, [`train.py:324-354`](../train.py#L324-L354)) so pre-trained weights aren't overwritten. |
| **15% Department UNK Dropout** (`32-dim`) | [`src/model.py:194-205`](../src/model.py#L194-L205)<br>[`src/model.py:256-262`](../src/model.py#L256-L262) | Trains the `UNK` (`0`) embedding so missing/unseen departments work at inference; prevents over-relying on department codes. | Slightly adds noise during training; unseen departments at test time share a single generic `UNK` representation. |
| **Piecewise-Linear Encoding (PLE)** (`2 -> 18 -> 64-dim`) | [`src/model.py:67-93`](../src/model.py#L67-L93)<br>[`src/multimodal_dataset.py:512-547`](../src/multimodal_dataset.py#L512-L547) | Monotone "thermometer" ramps guarantee the OPEX $\rightarrow$ CAPEX switch only flips once as price rises (0% non-monotone wiggles). | Data-learned bin edges place the switch within ~0.6% of €800 rather than at €800.00 to the exact cent. |
| **FiLM Gating** (`γ, β`) | [`src/model.py:199-202`](../src/model.py#L199-L202)<br>[`src/model.py:264-272`](../src/model.py#L264-L272) | Prevents the 768-dim text vector from drowning out the 64-dim price and 32-dim department; exposes live modulation diagnostics. | Adds ~147k parameters and couples the text representation with the tabular inputs before final classification. |
| **Counterfactual Augmentation + `1/5` Sample Weighting** | [`src/multimodal_dataset.py:320-327`](../src/multimodal_dataset.py#L320-L327)<br>[`src/multimodal_dataset.py:469-498`](../src/multimodal_dataset.py#L469-L498) | Teaches the exact €800 flip by showing the same IT description above and below €800, while `0.2` weights keep class proportions realistic. | Roughly doubles the number of training rows (659 originals + 644 counterfactuals = 1,303 rows) and requires rule access at dataset build time. |
| **Confidence Routing (`0.90` cutoff + `0.05` label smoothing)** | [`src/model.py:308-329`](../src/model.py#L308-L329)<br>[`classify_erp.py:219-223`](../classify_erp.py#L219-L223) | Auto-posts 94.3% of test invoices with 100% precision while catching €800 borderline cases and €0 credit notes for manual review. | ~5.7% of invoices require a human click instead of 100% hands-off posting. |
