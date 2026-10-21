"""
Multimodal Dataset Builder for ERP GL Account Classification.

Extracts and normalizes:
- Text features: vendor + description (no hardcoded rules)
- Continuous features: net_amount, log1p(net_amount) with StandardScaler
- Categorical features: department ID

Key principle: NO business logic or GL classification rules here.
The model learns thresholds (800 EUR) via backpropagation, not hardcoding.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.table import Table

from .config import GLAccount, LABEL2ID

console = Console()


@dataclass
class MultimodalFeatures:
    """Container for all modalities."""
    text: str
    num_features: np.ndarray  # [net_amount, log1p(net_amount)]
    dept_id: int
    label: int
    metadata: Dict


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


class MultimodalDatasetBuilder:
    """
    Build multimodal features WITHOUT hardcoded business rules.

    The model learns:
    - 800 EUR threshold from continuous features (not text preprocessing)
    - Department patterns from embeddings
    - Item semantics from text

    NO GL classification logic here!
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.dept_vocab = DepartmentVocabulary()
        self.fitted = False

    def extract_features(self, record: dict) -> MultimodalFeatures:
        """
        Extract multimodal features from a single record.

        Text: [VENDOR] {vendor} [DESC] {description}
        NO price or department in text → model must learn from numerical/categorical features
        """
        # Text: vendor + description only (no price, no dept)
        text = f"[VENDOR] {record['vendor']} [DESC] {record['description']}"

        # Continuous: net_amount and log-transformed amount
        # Log transform helps model learn both small (10 EUR) and large (2000 EUR) amounts
        net_amount = float(record['net_amount_eur'])
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
            metadata=record.get('metadata', {})
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
    ) -> Tuple[Path, Path, Path]:
        """
        Build train/val/test splits with multimodal features.

        Returns:
            (train_path, val_path, test_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract labels for stratification
        labels = [
            LABEL2ID[r['gl_account'].value if hasattr(r['gl_account'], 'value')
                     else str(r['gl_account'])]
            for r in records
        ]

        # Stratified split
        console.print(f"\n📂 Creating stratified splits from {len(records)} records...")

        # First split: train vs (val + test)
        train_records, temp_records, train_labels, temp_labels = train_test_split(
            records, labels,
            train_size=train_ratio,
            stratify=labels,
            random_state=self.random_seed
        )

        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
        val_records, test_records = train_test_split(
            temp_records,
            train_size=val_ratio_adjusted,
            stratify=temp_labels,
            random_state=self.random_seed
        )

        console.print(f"  Train: {len(train_records)} ({len(train_records)/len(records)*100:.1f}%)")
        console.print(f"  Val:   {len(val_records)} ({len(val_records)/len(records)*100:.1f}%)")
        console.print(f"  Test:  {len(test_records)} ({len(test_records)/len(records)*100:.1f}%)")

        # Fit on training data only
        self.fit(train_records)

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

        console.print(f"💾 Saved scaler to {scaler_path}")
        console.print(f"💾 Saved department vocabulary to {vocab_path}")

        # Print class distribution
        self._print_distribution(train_features, val_features, test_features)

        return train_path, val_path, test_path

    def _print_distribution(
        self,
        train_features: List[MultimodalFeatures],
        val_features: List[MultimodalFeatures],
        test_features: List[MultimodalFeatures]
    ):
        """Print GL account distribution across splits."""
        table = Table(title="\nGL Account Distribution Across Splits", show_header=True)
        table.add_column("GL Account", style="cyan")
        table.add_column("Train", justify="right", style="green")
        table.add_column("Val", justify="right", style="yellow")
        table.add_column("Test", justify="right", style="magenta")

        # Get ID to label mapping
        id2label = {v: k for k, v in LABEL2ID.items()}

        train_counts = Counter(f.label for f in train_features)
        val_counts = Counter(f.label for f in val_features)
        test_counts = Counter(f.label for f in test_features)

        all_labels = sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys()))

        for label_id in all_labels:
            label_str = id2label.get(label_id, f"UNKNOWN_{label_id}")
            table.add_row(
                label_str,
                str(train_counts.get(label_id, 0)),
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
