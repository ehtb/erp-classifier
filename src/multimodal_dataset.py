"""
Multimodal Dataset Builder for ERP GL Account Classification.

Extracts and normalizes:
- Text features: vendor + description (explicit price quotes scrubbed)
- Continuous features: net_amount, log1p(net_amount) with StandardScaler
- Categorical features: department ID

The GL rule itself lives in `src/config.py`. This module imports the rule only
to (a) discover amount boundaries, (b) keep near-duplicates together in the
split, and (c) generate rule-derived counterfactual amounts. It is therefore
rule *distillation*, not discovery: the boundary location is an input to the
training distribution, not something the model has to find from scratch.
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter

from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.table import Table

from .boundaries import amount_sensitive, counterfactual_amounts, near_boundary, price_band_edges
from .config import LABEL2ID, get_gl_account
from .text import PRICE_BAND_FILE, scrub_record, set_price_band

console = Console()


@dataclass
class MultimodalFeatures:
    """Container for all modalities."""
    text: str
    num_features: np.ndarray  # [net_amount, log1p(net_amount)]
    dept_id: int
    label: int
    metadata: Dict
    sample_weight: float = 1.0


class DepartmentVocabulary:
    """
    Department string → integer mapping.
    Fitted on training data only to prevent data leakage.
    """

    def __init__(self):
        self.dept2id = {"UNK": 0}  # Reserve 0 for unknown
        self.id2dept = {0: "UNK"}

    def fit(self, departments: List[str]):
        """Build vocabulary from training departments."""
        unique_depts = sorted(set(departments))
        for idx, dept in enumerate(unique_depts, start=1):
            self.dept2id[dept] = idx
            self.id2dept[idx] = dept

        console.print(f"✓ Department vocabulary: {len(self.dept2id)} unique values")
        console.print(f"  Departments: {list(self.dept2id.keys())}")

    def transform(self, department: str) -> int:
        """Map department to ID, returning UNK for unseen values."""
        return self.dept2id.get(department, 0)

    def save(self, path: Path):
        """Save vocabulary to JSON."""
        with open(path, 'w') as f:
            json.dump({
                'dept2id': self.dept2id,
                'id2dept': self.id2dept,
            }, f, indent=2)

    @classmethod
    def load(cls, path: Path):
        """Load vocabulary from JSON."""
        with open(path) as f:
            data = json.load(f)
        vocab = cls()
        vocab.dept2id = data['dept2id']
        # JSON keys are strings, convert back to int
        vocab.id2dept = {int(k): v for k, v in data['id2dept'].items()}
        return vocab


_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(record: dict) -> frozenset:
    """Alphanumeric tokens of the (uncorrupted) title, falling back to the description."""
    text = record.get('title') or record['description']
    return frozenset(t for t in _WORD_RE.findall(text.lower()) if len(t) > 1)


def near_duplicate_groups(records: List[dict], threshold: float = 0.5, max_df: float = 0.05) -> List[int]:
    """
    Group records whose titles have token-Jaccard >= threshold (union-find).
    Candidate pairs come from an inverted index over rare tokens, so cost stays
    near-linear instead of quadratic. Splitting by group keeps product-family
    near-duplicates out of both train and test.
    """
    toks = [_tokens(r) for r in records]
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    postings: Dict[str, List[int]] = {}
    for i, t in enumerate(toks):
        for w in t:
            postings.setdefault(w, []).append(i)

    limit = max(2, int(max_df * len(records)))
    for i, t in enumerate(toks):
        candidates = set()
        for w in t:
            if len(postings[w]) <= limit:
                candidates.update(postings[w])
        for j in candidates:
            if j > i:
                u = toks[j]
                if len(t & u) / len(t | u) >= threshold:
                    parent[find(i)] = find(j)

    return [find(i) for i in range(len(records))]


def stratum(record: dict) -> str:
    """
    Stratify on label x "near a rule boundary", where the boundary comes from the
    accounting rules themselves, so every split contains decision-boundary cases.
    """
    gl = record['gl_account']
    gl = gl.value if hasattr(gl, 'value') else str(gl)
    near = near_boundary(record.get('base_category'), float(record['net_amount_eur']))
    return f"{gl}|{'boundary' if near else 'other'}"


class MultimodalDatasetBuilder:
    """
    Build multimodal features WITHOUT hardcoded business rules.

    The model learns:
    - 800 EUR threshold from continuous features (not text preprocessing)
    - Department patterns from embeddings
    - Item semantics from text

    NO GL classification logic here, with one deliberate, opt-in exception: when
    `price_band_edges` is set (build_splits(price_band=True)), the text gets a
    rule-derived `[PRICE_BAND] n` token. That hands the model the rule boundary, so
    it is a with-prior upper bound (config F), never the default.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.dept_vocab = DepartmentVocabulary()
        self.fitted = False
        self.price_band_edges = None  # set by build_splits(price_band=True)

    def extract_features(self, record: dict) -> MultimodalFeatures:
        """
        Extract multimodal features from a single record.

        Text: [VENDOR] {vendor} [DESC] {description}
        NO price or department in text → model must learn from numerical/categorical features
        (except the opt-in `[PRICE_BAND] n` token, see the class docstring)
        """
        # Text: vendor + description only (no price, no dept)
        text = f"[VENDOR] {record['vendor']} [DESC] {record['description']}"

        # Continuous: net_amount and log-transformed amount
        # Log transform helps model learn both small (10 EUR) and large (2000 EUR) amounts
        net_amount = float(record['net_amount_eur'])

        # Two-stage model: stage 1 (rule-derived) labels the price band, stage 2 reads
        # it as a text token. Counterfactual copies get the token for their own amount.
        if self.price_band_edges is not None:
            text = set_price_band(text, net_amount, self.price_band_edges)
        num_features = np.array([net_amount, np.log1p(net_amount)], dtype=np.float32)

        # Categorical: department
        dept_id = self.dept_vocab.transform(record['department'])

        # Label: GL account
        gl_account = record['gl_account']
        label_str = gl_account.value if hasattr(gl_account, 'value') else str(gl_account)
        label = LABEL2ID[label_str]

        return MultimodalFeatures(
            text=text,
            num_features=num_features,
            dept_id=dept_id,
            label=label,
            metadata={
                'net_amount_eur': net_amount,
                'base_category': record.get('base_category'),
                'language': record.get('language'),
                'text_style': record.get('text_style'),
                'seed_id': record.get('seed_id'),
                'title': record.get('title'),
                'group_id': record.get('_group_id'),
                'price_quoted': bool(record.get('price_quoted', False)),
                'corrupted': record.get('corrupted', False),
                'counterfactual': record.get('counterfactual', False),
            },
            sample_weight=float(record.get('sample_weight', 1.0)),
        )

    def fit(self, train_records: List[dict]):
        """
        Fit scaler and vocabulary on training data ONLY.
        Critical: no data leakage from validation/test.
        """
        console.print("\n📊 Fitting scalers and vocabularies on training data...")

        # Fit department vocabulary
        departments = [r['department'] for r in train_records]
        self.dept_vocab.fit(departments)

        # Fit numerical scaler
        amounts = np.array([
            [float(r['net_amount_eur']), np.log1p(float(r['net_amount_eur']))]
            for r in train_records
        ], dtype=np.float32)

        self.scaler.fit(amounts)
        console.print(f"✓ Numerical scaler fitted")
        console.print(f"  Mean: {self.scaler.mean_}")
        console.print(f"  Std:  {self.scaler.scale_}")

        self.fitted = True

    def transform(self, records: List[dict]) -> List[MultimodalFeatures]:
        """
        Transform records to multimodal features.
        Scaler and vocabulary must be fitted first.
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")

        features_list = []
        for record in records:
            features = self.extract_features(record)

            # Normalize continuous features
            features.num_features = self.scaler.transform(
                features.num_features.reshape(1, -1)
            ).flatten().astype(np.float32)

            features_list.append(features)

        return features_list

    def build_splits(
        self,
        records: List[dict],
        output_dir: Path,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        counterfactuals_per_record: int = 0,
        bin_strategy: str = "quantile",
        bin_weighting: str = "unweighted",
        n_bins: int = 16,
        price_band: bool = False,
    ) -> Tuple[Path, Path, Path]:
        """
        Build train/val/test splits with multimodal features.

        Returns:
            (train_path, val_path, test_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.price_band_edges = list(price_band_edges()) if price_band else None

        # Scrub explicit price quotes before anything else: the description must not
        # be an alternative copy of the amount the model is supposed to read from the
        # numeric branch. The source flag is kept for reporting.
        records = [scrub_record(r) for r in records]
        n_quoted = sum(1 for r in records if r.get('price_quoted'))
        console.print(f"  Scrubbed explicit price quotes from {n_quoted} descriptions")

        # Grouped + stratified split: near-duplicate products stay in one split,
        # and cases near a rule boundary are spread across all splits.
        console.print(f"\n📂 Creating grouped, stratified splits from {len(records)} records...")
        groups = np.array(near_duplicate_groups(records))
        strata = np.array([stratum(r) for r in records])
        for i, group_id in enumerate(groups):
            records[i]['_group_id'] = int(group_id)
        console.print(f"  Near-duplicate groups: {len(set(groups))} (from {len(records)} records)")

        train_idx, val_idx, test_idx = self._greedy_group_split(
            records, groups, strata, train_ratio, val_ratio
        )

        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        test_records = [records[i] for i in test_idx]

        console.print(f"  Train: {len(train_records)} ({len(train_records)/len(records)*100:.1f}%)")
        console.print(f"  Val:   {len(val_records)} ({len(val_records)/len(records)*100:.1f}%)")
        console.print(f"  Test:  {len(test_records)} ({len(test_records)/len(records)*100:.1f}%)")

        # Fit on training data only
        self.fit(train_records)

        # Counterfactual augmentation (train split only): same text/department, amount
        # resampled around the rule boundaries, label re-derived from the accounting rules.
        # Each counterfactual carries weight 1/(k+1) so an original record plus its k
        # copies count once in the loss; this restores the source class prior instead
        # of letting the IT classes dominate by construction.
        if counterfactuals_per_record > 0:
            family_weight = 1.0 / (counterfactuals_per_record + 1)
            for rec in train_records:
                if amount_sensitive(rec.get('base_category')):
                    rec['sample_weight'] = family_weight
            train_records = train_records + self.counterfactuals(
                train_records, counterfactuals_per_record
            )

        # Transform all splits
        console.print("\n🔄 Transforming splits...")
        train_features = self.transform(train_records)
        val_features = self.transform(val_records)
        test_features = self.transform(test_records)

        # Save to JSONL
        train_path = output_dir / "train_multimodal.jsonl"
        val_path = output_dir / "val_multimodal.jsonl"
        test_path = output_dir / "test_multimodal.jsonl"

        for features_list, path in [
            (train_features, train_path),
            (val_features, val_path),
            (test_features, test_path),
        ]:
            with open(path, 'w') as f:
                for feat in features_list:
                    f.write(json.dumps({
                        'text': feat.text,
                        'num_features': feat.num_features.tolist(),
                        'dept_id': feat.dept_id,
                        'label': feat.label,
                        'sample_weight': feat.sample_weight,
                        'metadata': feat.metadata,
                    }) + '\n')

        console.print(f"\n💾 Saved multimodal splits to {output_dir}")

        # Save scaler and vocabulary for inference
        scaler_path = output_dir / "scaler.json"
        with open(scaler_path, 'w') as f:
            json.dump({
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist(),
            }, f, indent=2)

        vocab_path = output_dir / "dept_vocab.json"
        self.dept_vocab.save(vocab_path)

        bins_path = output_dir / "numeric_bins.json"
        edges = self.fit_bin_edges(train_features, bin_strategy, n_bins, weighting=bin_weighting)
        with open(bins_path, 'w') as f:
            json.dump({
                'strategy': bin_strategy,
                'weighting': bin_weighting,
                'feature_idx': 1,
                'edges': edges,
            }, f, indent=2)
        console.print(f"💾 Saved amount bin edges ({bin_strategy}, {len(edges) - 1} bins) to {bins_path}")

        band_path = output_dir / PRICE_BAND_FILE
        if self.price_band_edges is not None:
            with open(band_path, 'w') as f:
                json.dump({'edges': self.price_band_edges}, f, indent=2)
            console.print(f"💾 Saved price band edges {self.price_band_edges} to {band_path}")
        elif band_path.exists():
            band_path.unlink()  # stale file from an earlier two-stage build

        console.print(f"💾 Saved scaler to {scaler_path}")
        console.print(f"💾 Saved department vocabulary to {vocab_path}")

        # Print class distribution
        self._print_distribution(train_features, val_features, test_features)

        return train_path, val_path, test_path

    @staticmethod
    def _greedy_group_split(
        records: List[dict],
        groups: np.ndarray,
        strata: np.ndarray,
        train_ratio: float,
        val_ratio: float,
    ):
        """
        Assign whole near-duplicate groups to train/val/test.

        The old fold-concatenation approach was at the mercy of a few very large
        product-family groups (29-36 near-duplicates), which produced test class
        counts of 8 against 34 in validation. Greedy assignment processes groups
        largest-first and penalises the incremental deviation from the target class
        proportions, so large groups go where there is still capacity. The split is
        deterministic and independent of the model seed.
        """
        ratios = {
            'train': train_ratio,
            'val': val_ratio,
            'test': max(0.0, 1.0 - train_ratio - val_ratio),
        }
        n = len(records)
        target_size = {k: r * n for k, r in ratios.items()}
        totals = Counter(strata)
        target = {
            k: {s: ratios[k] * totals[s] for s in set(strata)}
            for k in ratios
        }

        members: Dict[int, List[int]] = {}
        for i, g in enumerate(groups):
            members.setdefault(int(g), []).append(i)

        # Largest groups first; group id breaks ties deterministically.
        order = sorted(members.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        counts = {k: Counter() for k in ratios}
        sizes = {k: 0 for k in ratios}
        chosen = {k: [] for k in ratios}

        for gid, idxs in order:
            group_counts = Counter(strata[i] for i in idxs)
            scores = {}
            for split in ratios:
                class_cost = 0.0
                for s, c in group_counts.items():
                    tgt = max(target[split][s], 1.0)
                    cur = counts[split][s]
                    class_cost += ((cur + c - tgt) ** 2 - (cur - tgt) ** 2) / tgt
                tgt_size = max(target_size[split], 1.0)
                cur_size = sizes[split]
                size_cost = (
                    (cur_size + len(idxs) - tgt_size) ** 2 - (cur_size - tgt_size) ** 2
                ) / tgt_size
                scores[split] = class_cost + 0.5 * size_cost
            split = min(scores, key=scores.get)
            chosen[split].extend(idxs)
            sizes[split] += len(idxs)
            for s, c in group_counts.items():
                counts[split][s] += c

        console.print(
            "  Grouped split sizes: "
            + ", ".join(f"{k}={sizes[k]}" for k in ('train', 'val', 'test'))
        )

        return (
            np.array(sorted(chosen['train']), dtype=int),
            np.array(sorted(chosen['val']), dtype=int),
            np.array(sorted(chosen['test']), dtype=int),
        )

    def counterfactuals(self, records: List[dict], per_record: int) -> List[dict]:
        """
        Re-amounted copies of amount-sensitive records. The GL label comes from the
        accounting rules (`get_gl_account`), never from a hard-coded threshold, so the
        model has to read the amount instead of leaning on text style that merely
        correlates with price.

        The original record and its copies are all down-weighted to 1/(per_record+1)
        by `build_splits`, so an augmented family has the same total loss weight as
        one unaugmented record. This keeps the source class prior instead of letting
        the IT classes dominate by construction.
        """
        rng = np.random.default_rng(self.random_seed)
        out = []
        for rec in records:
            cat = rec.get('base_category')
            if not amount_sensitive(cat):
                continue
            for amount in counterfactual_amounts(cat, per_record, rng):
                new = dict(rec)
                new['net_amount_eur'] = amount
                new['gl_account'] = get_gl_account(cat, amount).value
                new['counterfactual'] = True
                new['sample_weight'] = 1.0 / (per_record + 1)
                out.append(new)
        console.print(
            f"✓ Added {len(out)} counterfactual samples (train only, weight "
            f"{1.0/(per_record+1):.2f} each)"
        )
        return out

    @staticmethod
    def _weighted_quantile(x: np.ndarray, q: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Quantiles of `x` with observation weights `w`."""
        order = np.argsort(x)
        x, w = x[order], w[order]
        cumulative = np.cumsum(w)
        if cumulative[-1] <= 0:
            return np.quantile(x, q)
        cumulative = cumulative / cumulative[-1]
        return np.interp(q, cumulative, x)

    @staticmethod
    def fit_bin_edges(
        train_features: List[MultimodalFeatures],
        strategy: str,
        n_bins: int,
        weighting: str = "unweighted",
    ) -> List[float]:
        """
        Bin edges over the (standardised) log-amount for piecewise-linear encoding,
        learned from training data only.

        quantile: equal-mass bins. The default is intentionally *unweighted* over
                  the augmented set, so resolution is allocated around the
                  counterfactual boundary. `weighting="weighted"` uses the family
                  weights and therefore follows the source prior instead.
        tree:     target-aware edges from a shallow decision tree. The tree always
                  receives the family weights; otherwise the 4x counterfactual IT
                  copies dominate its split criterion.
        """
        if weighting not in ("unweighted", "weighted"):
            raise ValueError("weighting must be 'unweighted' or 'weighted'")
        x = np.array([f.num_features[1] for f in train_features], dtype=np.float64)
        w = np.array([float(f.sample_weight) for f in train_features], dtype=np.float64)
        if strategy == "tree":
            from sklearn.tree import DecisionTreeClassifier
            y = np.array([f.label for f in train_features])
            tree = DecisionTreeClassifier(max_leaf_nodes=n_bins, min_samples_leaf=5, random_state=0)
            tree.fit(x.reshape(-1, 1), y, sample_weight=w)
            inner = tree.tree_.threshold[tree.tree_.feature >= 0]
        else:
            q = np.linspace(0, 1, n_bins + 1)[1:-1]
            inner = (
                MultimodalDatasetBuilder._weighted_quantile(x, q, w)
                if weighting == "weighted" else np.quantile(x, q)
            )
        edges = np.unique(np.concatenate([[x.min()], inner, [x.max()]]))
        return [float(e) for e in edges]

    def _print_distribution(
        self,
        train_features: List[MultimodalFeatures],
        val_features: List[MultimodalFeatures],
        test_features: List[MultimodalFeatures]
    ):
        """Print GL account distribution across splits."""
        table = Table(title="\nGL Account Distribution Across Splits", show_header=True)
        table.add_column("GL Account", style="cyan")
        table.add_column("Train (n)", justify="right", style="green")
        table.add_column("Train (eff.)", justify="right", style="green")
        table.add_column("Val", justify="right", style="yellow")
        table.add_column("Test", justify="right", style="magenta")

        # Get ID to label mapping
        id2label = {v: k for k, v in LABEL2ID.items()}

        train_counts = Counter(f.label for f in train_features)
        train_eff = Counter()
        for f in train_features:
            train_eff[f.label] += float(f.sample_weight)
        val_counts = Counter(f.label for f in val_features)
        test_counts = Counter(f.label for f in test_features)

        all_labels = sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys()))

        for label_id in all_labels:
            label_str = id2label.get(label_id, f"UNKNOWN_{label_id}")
            table.add_row(
                label_str,
                str(train_counts.get(label_id, 0)),
                f"{train_eff.get(label_id, 0.0):.1f}",
                str(val_counts.get(label_id, 0)),
                str(test_counts.get(label_id, 0)),
            )

        console.print(table)


def load_multimodal_dataset(jsonl_path: Path) -> List[Dict]:
    """Load multimodal features from JSONL."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            # Convert lists back to numpy arrays
            record['num_features'] = np.array(record['num_features'], dtype=np.float32)
            records.append(record)
    return records
