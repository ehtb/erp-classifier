#!/usr/bin/env python3
"""
End-to-end data generation pipeline for ERP GL classifier.

What this does:
1. Generate seeds - combinatorial specs (vendor, dept, price, language, style)
2. Verbalize - LLM generates realistic Amazon item descriptions from specs
3. Corrupt - add SAP truncation, PO prefixes, OCR errors (real ERP noise)
4. Filter - critic verifies the GL label matches what an LLM would predict zero-shot

After data generation:
- Use build_multimodal_dataset.py to convert to multimodal format
- Use train_multimodal.py to train the classifier
- Use evaluate_multimodal.py to evaluate performance
"""

import argparse
import json
from pathlib import Path
import sys
import os

from rich.console import Console
from rich.panel import Panel

from src.seed_generator import SeedGenerator, print_distribution_summary
from src.verbalizer import Verbalizer
from src.corruptor import ERPCorruptor
from src.critic import Critic
from src.dataset_builder import DatasetBuilder
from src.config import DATA_GEN_CONFIG

console = Console()


def run_pipeline(
    num_samples: int,
    output_dir: Path,
    dry_run: bool = False,
    llm_backend: str = "openai",
    llm_model: str = None,
):
    """
    Run the data generation pipeline.

    Args:
        num_samples: Number of samples to generate
        output_dir: Base output directory
        dry_run: Run with minimal samples for testing
        llm_backend: LLM backend to use (openai, openrouter, bedrock)
        llm_model: Model name/ID
    """
    if dry_run:
        num_samples = 160
        console.print("[yellow]🧪 Dry run mode: using 160 samples[/yellow]\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    raw_data_path = output_dir / "01_verbalized.jsonl"
    corrupted_data_path = output_dir / "02_corrupted.jsonl"
    filtered_data_path = output_dir / "03_filtered.jsonl"
    rejected_data_path = output_dir / "03_rejected.jsonl"
    splits_dir = output_dir / "splits"
    model_dir = output_dir / "model"

    console.print(Panel.fit(
        "[bold cyan]ERP GL Account Classifier Pipeline[/bold cyan]\n\n"
        f"Samples: {num_samples}\n"
        f"Output: {output_dir}",
        title="Pipeline Configuration"
    ))

    # Step 1: Generate seeds
    console.print("\n[bold cyan]Step 1/7: Generate Seeds[/bold cyan]")
    console.print("=" * 60)

    generator = SeedGenerator(seed=DATA_GEN_CONFIG["random_seed"])
    seeds = generator.generate_seeds(num_samples)

    console.print(f"\n✓ Generated {len(seeds)} seeds")
    print_distribution_summary(seeds)

    # Generate counterfactual pairs
    num_cf_pairs = 20 if dry_run else DATA_GEN_CONFIG["counterfactual_pairs"]
    cf_pairs = generator.generate_counterfactual_pairs(num_cf_pairs)
    console.print(f"✓ Generated {len(cf_pairs)} counterfactual pairs")

    # Step 2: Verbalize with LLM
    console.print("\n[bold cyan]Step 2/7: Verbalize with LLM[/bold cyan]")
    console.print("=" * 60)

    # Initialize verbalizer with specified backend
    llm_kwargs = {}
    if llm_model:
        if llm_backend == "bedrock":
            llm_kwargs["model_id"] = llm_model
        else:  # openai, openrouter
            llm_kwargs["model"] = llm_model

    verbalizer = Verbalizer(backend=llm_backend, **llm_kwargs)

    console.print(f"\n🤖 Verbalizing {len(seeds)} seeds with {llm_backend}...")
    verbalized_records = verbalizer.verbalize_batch(seeds, show_progress=True)

    # Save verbalized data
    with open(raw_data_path, 'w') as f:
        for record in verbalized_records:
            f.write(json.dumps(record) + '\n')

    console.print(f"✓ Saved {len(verbalized_records)} verbalized records to {raw_data_path}")

    # Verbalize counterfactual pairs
    console.print(f"\n🤖 Verbalizing {len(cf_pairs)} counterfactual pairs...")
    cf_records = []
    for low, high in cf_pairs:
        low_v = verbalizer.verbalize_one(low)
        high_v = verbalizer.verbalize_one(high)
        # Use same description for both (counterfactual condition)
        high_v['description'] = low_v['description']
        cf_records.extend([low_v, high_v])

    # Step 3: Apply corruption
    console.print("\n[bold cyan]Step 3/7: Apply ERP Corruption[/bold cyan]")
    console.print("=" * 60)

    corruptor = ERPCorruptor(
        corruption_ratio=DATA_GEN_CONFIG["corruption_ratio"],
        seed=DATA_GEN_CONFIG["random_seed"],
    )

    corrupted_records = corruptor.corrupt_batch(verbalized_records)

    with open(corrupted_data_path, 'w') as f:
        for record in corrupted_records:
            f.write(json.dumps(record) + '\n')

    num_corrupted = sum(1 for r in corrupted_records if r.get('corrupted', False))
    console.print(f"✓ Applied corruption to {num_corrupted}/{len(corrupted_records)} records")
    console.print(f"✓ Saved to {corrupted_data_path}")

    # Step 4: Filter with critic
    console.print("\n[bold cyan]Step 4/7: Filter with Critic[/bold cyan]")
    console.print("=" * 60)

    critic = Critic(backend=llm_backend, **llm_kwargs)

    console.print(f"\n🔍 Verifying {len(corrupted_records)} records with {llm_backend}...")
    verified_records = critic.verify_batch(corrupted_records, show_progress=True)

    passed, rejected = critic.filter_verified(verified_records)

    with open(filtered_data_path, 'w') as f:
        for record in passed:
            f.write(json.dumps(record) + '\n')

    with open(rejected_data_path, 'w') as f:
        for record in rejected:
            f.write(json.dumps(record) + '\n')

    console.print(f"✓ Passed: {len(passed)}/{len(verified_records)} ({len(passed)/len(verified_records)*100:.1f}%)")
    console.print(f"✓ Saved to {filtered_data_path}")

    # Step 5: Build dataset splits
    console.print("\n[bold cyan]Step 5/7: Build Dataset Splits[/bold cyan]")
    console.print("=" * 60)

    builder = DatasetBuilder(
        train_ratio=DATA_GEN_CONFIG["train_ratio"],
        val_ratio=DATA_GEN_CONFIG["val_ratio"],
        test_ratio=DATA_GEN_CONFIG["test_ratio"],
        random_seed=DATA_GEN_CONFIG["random_seed"],
    )

    splits_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = builder.build_splits(passed, splits_dir)

    # Build counterfactual test set
    cf_pairs_tuples = [(cf_records[i], cf_records[i+1]) for i in range(0, len(cf_records), 2)]
    cf_dataset_path = builder.build_counterfactual_test(cf_pairs_tuples, splits_dir)

    console.print(f"✓ Saved counterfactual test set to {cf_dataset_path}")

    # Summary
    console.print("\n" + "=" * 60)
    console.print(Panel.fit(
        "[bold green]✓ Pipeline Complete![/bold green]\n\n"
        f"Generated samples: {len(seeds)}\n"
        f"Verbalized: {len(verbalized_records)}\n"
        f"Passed critic: {len(passed)}\n"
        f"Output directory: {output_dir}",
        title="Pipeline Summary"
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end ERP GL classifier pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=DATA_GEN_CONFIG["test_samples"],
        help="Number of samples to generate (default: 160 for testing, use 10000 for full run)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick test run with 160 samples",
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="openai",
        choices=["openai", "openrouter", "bedrock"],
        help="LLM backend (default: openai)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        help="LLM model name/ID (backend-specific, uses defaults if not specified)",
    )

    args = parser.parse_args()

    # Check backend-specific requirements
    if args.llm_backend == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            console.print("[red]Error: OPENAI_API_KEY not set[/red]")
            console.print("Set it with: export OPENAI_API_KEY=sk-...")
            sys.exit(1)
        console.print("[cyan]Using OpenAI API[/cyan]")
    elif args.llm_backend == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            console.print("[red]Error: OPENROUTER_API_KEY not set[/red]")
            console.print("Set it with: export OPENROUTER_API_KEY=sk-or-...")
            sys.exit(1)
        console.print("[cyan]Using OpenRouter API[/cyan]")
    elif args.llm_backend == "bedrock":
        if not os.getenv("AWS_REGION"):
            console.print("[yellow]Warning: AWS_REGION not set, using default us-east-1[/yellow]")
        console.print(f"[cyan]Using AWS Bedrock in {os.getenv('AWS_REGION', 'us-east-1')}[/cyan]")

    # Run pipeline
    run_pipeline(
        num_samples=args.num_samples,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
    )


if __name__ == "__main__":
    main()
