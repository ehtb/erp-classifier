"""
Dataset builder for creating stratified train/val/test splits.

Creates:
1. Stratified train/val/test splits (70/15/15 default)
2. Counterfactual test set for robustness evaluation
3. HuggingFace Dataset format with Arrow/Parquet backend
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

from rich.console import Console
from rich.table import Table
import sklearn.model_selection as model_selection
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Sequence

from .config import DATA_GEN_CONFIG, GLAccount

console = Console()


def format_for_training(record: dict) -> dict:
    """Format record for training (extract text and label)."""
    # Format as ERP invoice line
    text = (
        f"[VENDOR] {record['vendor']} "
        f"[DEPT] {record['department']} "
        f"[NET_AMT] {record['net_amount_eur']:.2f} "
        f"[CURR] EUR "
        f"[DESC] {record['description']}"
    )

    # Extract string value from gl_account (handles both string and enum)
    gl_account = record['gl_account']
    label = gl_account.value if hasattr(gl_account, 'value') else str(gl_account)

    return {
        "text": text,
        "label": label,
        "metadata": {
            "seed_id": int(record.get('seed_id') or -1),  # int64 rejects None
            "base_category": record.get('base_category', ''),
            "department": record.get('department', ''),
            "net_amount_eur": float(record.get('net_amount_eur') or 0.0),
            "corrupted": bool(record.get('corrupted', False)),
            "corruption_types": record.get('corruption_types', []),
        }
    }


class DatasetBuilder:
    """Build stratified train/val/test splits with counterfactuals."""

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        """
        Initialize dataset builder.

        Args:
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            random_seed: Random seed for reproducibility
        """
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)

        # HuggingFace Dataset features schema
        self.features = Features({
            "text": Value("string"),
            "label": ClassLabel(names=[gl.value for gl in GLAccount]),
            "metadata": {
                "seed_id": Value("int64"),
                "base_category": Value("string"),
                "department": Value("string"),
                "net_amount_eur": Value("float64"),
                "corrupted": Value("bool"),
                "corruption_types": Sequence(Value("string")),
            }
        })

    def stratified_split(
        self,
        records: List[dict]
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        Create stratified train/val/test split.

        Ensures balanced GL account distribution in each split.
        Falls back to non-stratified split if classes are too small.
        """
        # Extract labels for stratification
        labels = [r['gl_account'] for r in records]

        # Check if stratification is possible
        label_counts = Counter(labels)
        min_samples = min(label_counts.values())

        # Need at least 2 samples per class for stratification
        # (sklearn requires n_splits=2 minimum for stratification)
        can_stratify = min_samples >= 2

        if not can_stratify:
            console.print(
                f"\n[yellow]⚠ Warning: Some classes have only {min_samples} sample(s). "
                f"Using random split instead of stratified.[/yellow]"
            )
            console.print(f"[yellow]  Class distribution: {dict(label_counts)}[/yellow]\n")

        # First split: train vs (val + test)
        try:
            train_records, temp_records = model_selection.train_test_split(
                records,
                train_size=self.train_ratio,
                stratify=labels if can_stratify else None,
                random_state=self.random_seed,
            )
        except ValueError as e:
            # Fallback if stratification still fails
            console.print(f"[yellow]⚠ Stratification failed: {e}[/yellow]")
            console.print("[yellow]  Using random split...[/yellow]\n")
            train_records, temp_records = model_selection.train_test_split(
                records,
                train_size=self.train_ratio,
                stratify=None,
                random_state=self.random_seed,
            )
            can_stratify = False

        # Second split: val vs test
        temp_labels = [r['gl_account'] for r in temp_records]
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)

        try:
            val_records, test_records = model_selection.train_test_split(
                temp_records,
                train_size=val_ratio_adjusted,
                stratify=temp_labels if can_stratify else None,
                random_state=self.random_seed,
            )
        except ValueError as e:
            console.print(f"[yellow]⚠ Stratification failed on second split: {e}[/yellow]")
            console.print("[yellow]  Using random split...[/yellow]\n")
            val_records, test_records = model_selection.train_test_split(
                temp_records,
                train_size=val_ratio_adjusted,
                stratify=None,
                random_state=self.random_seed,
            )

        return train_records, val_records, test_records

    def build_splits(
        self,
        records: List[dict],
        output_dir: Path,
    ) -> Path:
        """
        Build and save train/val/test splits in HuggingFace Dataset format.

        Args:
            records: List of verified records
            output_dir: Output directory for splits

        Returns:
            Path to saved HuggingFace DatasetDict
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create stratified splits
        train, val, test = self.stratified_split(records)

        console.print(f"\n[bold]Split Statistics:[/bold]")
        console.print(f"  Train: {len(train)} ({len(train)/len(records)*100:.1f}%)")
        console.print(f"  Val:   {len(val)} ({len(val)/len(records)*100:.1f}%)")
        console.print(f"  Test:  {len(test)} ({len(test)/len(records)*100:.1f}%)")

        # Check balance
        self._print_split_balance(train, val, test)

        # Format for training
        splits = {
            "train": [format_for_training(r) for r in train],
            "val": [format_for_training(r) for r in val],
            "test": [format_for_training(r) for r in test],
        }

        # Save to HuggingFace format
        hf_dataset = self._build_hf_dataset(splits)
        hf_dir = output_dir / "dataset"
        hf_dataset.save_to_disk(str(hf_dir))
        console.print(f"💾 Saved HuggingFace dataset to {hf_dir}")
        console.print(f"   Train: {len(hf_dataset['train'])} samples")
        console.print(f"   Validation: {len(hf_dataset['validation'])} samples")
        console.print(f"   Test: {len(hf_dataset['test'])} samples")

        return hf_dir

    def _print_split_balance(
        self,
        train: List[dict],
        val: List[dict],
        test: List[dict]
    ) -> None:
        """Print GL account balance across splits."""
        table = Table(title="\nGL Account Distribution Across Splits", show_header=True)
        table.add_column("GL Account", style="cyan")
        table.add_column("Train", justify="right", style="green")
        table.add_column("Val", justify="right", style="yellow")
        table.add_column("Test", justify="right", style="magenta")

        train_counts = Counter(r['gl_account'] for r in train)
        val_counts = Counter(r['gl_account'] for r in val)
        test_counts = Counter(r['gl_account'] for r in test)

        all_gls = sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys()))

        for gl in all_gls:
            table.add_row(
                gl,
                str(train_counts.get(gl, 0)),
                str(val_counts.get(gl, 0)),
                str(test_counts.get(gl, 0)),
            )

        console.print(table)

    def _build_hf_dataset(self, splits: Dict[str, List[dict]]) -> DatasetDict:
        """
        Build HuggingFace DatasetDict from split data.

        Args:
            splits: Dict with 'train', 'val', 'test' keys mapping to formatted records

        Returns:
            DatasetDict with train/validation/test splits
        """
        hf_splits = {}

        for split_name in ['train', 'val', 'test']:
            if split_name not in splits:
                continue

            split_data = splits[split_name]

            # Convert to HF-compatible format
            # metadata must be a list of dicts, not dict of lists
            hf_data = {
                'text': [r['text'] for r in split_data],
                'label': [r['label'] for r in split_data],
                'metadata': [r['metadata'] for r in split_data],
            }

            # Create Dataset
            hf_dataset = Dataset.from_dict(hf_data, features=self.features)

            # Map split names for HF convention (val -> validation)
            hf_split_name = 'validation' if split_name == 'val' else split_name
            hf_splits[hf_split_name] = hf_dataset

        return DatasetDict(hf_splits)

    def build_counterfactual_test(
        self,
        counterfactual_pairs: List[Tuple[dict, dict]],
        output_dir: Path,
    ) -> Path:
        """
        Build counterfactual test set from pairs in HuggingFace Dataset format.

        Each pair contains two records that differ only in price or department.

        Args:
            counterfactual_pairs: List of (low, high) record pairs
            output_dir: Output directory

        Returns:
            Path to saved HuggingFace Dataset
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Flatten pairs into records with pair_id
        test_records = []
        for pair_id, (low, high) in enumerate(counterfactual_pairs):
            low_formatted = format_for_training(low)
            low_formatted['pair_id'] = pair_id
            low_formatted['pair_position'] = 'low'

            high_formatted = format_for_training(high)
            high_formatted['pair_id'] = pair_id
            high_formatted['pair_position'] = 'high'

            test_records.extend([low_formatted, high_formatted])

        # Save HuggingFace format
        hf_path = self._save_counterfactuals_hf(test_records, output_dir)

        console.print(f"\n💾 Saved {len(test_records)} counterfactual test records to {hf_path}")
        console.print(f"   ({len(counterfactual_pairs)} pairs)")

        return hf_path

    def _save_counterfactuals_hf(
        self,
        test_records: List[dict],
        output_dir: Path,
    ) -> Path:
        """Save counterfactual test set as HuggingFace Dataset."""
        hf_data = {
            'text': [r['text'] for r in test_records],
            'label': [r['label'] for r in test_records],
            'pair_id': [r.get('pair_id', -1) for r in test_records],
            'pair_position': [r.get('pair_position', '') for r in test_records],
            'metadata': [r['metadata'] for r in test_records],
        }

        # Create Dataset with extended features
        features = Features({
            'text': Value('string'),
            'label': ClassLabel(names=[gl.value for gl in GLAccount]),
            'pair_id': Value('int64'),
            'pair_position': Value('string'),
            'metadata': {
                'seed_id': Value('int64'),
                'base_category': Value('string'),
                'department': Value('string'),
                'net_amount_eur': Value('float64'),
                'corrupted': Value('bool'),
                'corruption_types': Sequence(Value('string')),
            }
        })

        hf_dataset = Dataset.from_dict(hf_data, features=features)
        hf_path = output_dir / "hf_counterfactuals"
        hf_dataset.save_to_disk(str(hf_path))

        console.print(f"💾 Saved HuggingFace counterfactual dataset to {hf_path}")
        return hf_path


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Build train/val/test splits")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with filtered records",
    )
    parser.add_argument(
        "--counterfactuals",
        type=str,
        help="Input JSON file with counterfactual pairs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for splits",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DATA_GEN_CONFIG["train_ratio"],
        help="Train split ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DATA_GEN_CONFIG["val_ratio"],
        help="Validation split ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DATA_GEN_CONFIG["test_ratio"],
        help="Test split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DATA_GEN_CONFIG["random_seed"],
        help="Random seed",
    )

    args = parser.parse_args()

    # Load records
    console.print(f"📂 Loading records from {args.input}")
    records = []
    with open(args.input, 'r') as f:
        for line in f:
            records.append(json.loads(line))

    console.print(f"   Loaded {len(records)} records")

    # Build splits
    builder = DatasetBuilder(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )

    dataset_path = builder.build_splits(records, Path(args.output_dir))

    # Build counterfactual test set if provided
    if args.counterfactuals:
        console.print(f"\n📂 Loading counterfactual pairs from {args.counterfactuals}")
        with open(args.counterfactuals, 'r') as f:
            data = json.load(f)

        # Convert pairs back to tuples
        pairs = [(p['low'], p['high']) for p in data['counterfactual_pairs']]

        # Note: Pairs need to be verbalized first
        console.print(f"[yellow]⚠ Counterfactual pairs need to be verbalized before building test set[/yellow]")
        console.print(f"   Use: python -m src.verbalizer --input {args.counterfactuals} --output <verbalized.jsonl>")

    console.print(f"\n[bold green]✓ HuggingFace dataset created: {dataset_path}[/bold green]")
    console.print(f"\nLoad with:")
    console.print(f"  from datasets import load_from_disk")
    console.print(f"  dataset = load_from_disk('{dataset_path}')")


if __name__ == "__main__":
    main()
