"""
Build multimodal dataset from filtered records.

Usage:
    python build_multimodal_dataset.py \
        --input output/03_filtered.jsonl \
        --output-dir output/multimodal
"""

import argparse
import json
from pathlib import Path

from src.multimodal_dataset import MultimodalDatasetBuilder
from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Build multimodal ERP dataset")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with filtered records"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/multimodal",
        help="Output directory for multimodal splits"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    console.print(f"\n[bold]Building Multimodal Dataset[/bold]")
    console.print(f"  Input: {args.input}")
    console.print(f"  Output: {args.output_dir}")

    # Load records
    console.print(f"\n📂 Loading records from {args.input}...")
    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))

    console.print(f"✓ Loaded {len(records)} records")

    # Build multimodal splits
    builder = MultimodalDatasetBuilder(random_seed=args.seed)

    train_path, val_path, test_path = builder.build_splits(
        records=records,
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    console.print(f"\n[bold green]✓ Multimodal dataset created successfully![/bold green]")
    console.print(f"\n  Train: {train_path}")
    console.print(f"  Val:   {val_path}")
    console.print(f"  Test:  {test_path}")
    console.print(f"\n  Scaler: {Path(args.output_dir) / 'scaler.json'}")
    console.print(f"  Dept vocab: {Path(args.output_dir) / 'dept_vocab.json'}")


if __name__ == "__main__":
    main()
