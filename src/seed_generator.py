"""
Combinatorial seed generator for ERP GL classification training data.

Generates structured seed specifications across:
- Vendor variants
- Departments
- Languages
- Text styles
- Price ranges (with oversampling around IT hardware 800 EUR threshold)
- Base categories

Seeds serve as input to the LLM verbalizer which generates realistic descriptions.
"""

import random
from dataclasses import dataclass
from typing import List
import numpy as np
from collections import Counter

from .config import (
    GLAccount,
    Department,
    Language,
    TextStyle,
    VENDOR_VARIANTS,
    CATEGORY_DEPT_AFFINITY,
    PRICE_RANGES,
    get_gl_account,
    get_price_distribution_weights,
    DATA_GEN_CONFIG,
)


@dataclass
class SeedSpec:
    """Structured specification for a single training example."""

    seed_id: int
    base_category: str
    department: Department
    vendor: str
    language: Language
    text_style: TextStyle
    net_amount_eur: float
    gl_account: GLAccount

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "seed_id": self.seed_id,
            "base_category": self.base_category,
            "department": self.department.value,
            "vendor": self.vendor,
            "language": self.language.value,
            "text_style": self.text_style.value,
            "net_amount_eur": round(self.net_amount_eur, 2),
            "gl_account": self.gl_account.value,
        }


class SeedGenerator:
    """Generates combinatorial seed specifications for synthetic data."""

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def _sample_price(self, category: str) -> float:
        """
        Sample price with strategic oversampling.

        For IT_HARDWARE: sample more often between 650-950 EUR so we get enough training
        examples near the 800 EUR threshold. Without this, random sampling would give us
        too few boundary cases to verify the model learned the rule.

        For other categories: sample uniformly across the price range.
        """
        min_price, max_price = PRICE_RANGES[category]

        if category == "IT_HARDWARE":
            # Divide the price range into bins and weight each bin. The bins near 800 EUR
            # get 5x weight, so we sample from them more often.
            num_bins = 20
            weights = get_price_distribution_weights(category, num_bins)
            weights_array = np.array(weights, dtype=float)
            weights_array /= weights_array.sum()

            # Pick a bin according to the weights
            bin_idx = self.np_rng.choice(num_bins, p=weights_array)

            # Pick a price uniformly within that bin
            bin_size = (max_price - min_price) / num_bins
            bin_start = min_price + bin_idx * bin_size
            bin_end = bin_start + bin_size

            price = self.np_rng.uniform(bin_start, bin_end)
        else:
            # Simple uniform sampling for everything else
            price = self.np_rng.uniform(min_price, max_price)

        return round(price, 2)

    def _sample_department(self, category: str) -> Department:
        """Sample department based on category affinity."""
        if category in CATEGORY_DEPT_AFFINITY:
            eligible_depts = CATEGORY_DEPT_AFFINITY[category]
        else:
            # Fallback to all departments
            eligible_depts = list(Department)

        return self.rng.choice(eligible_depts)

    def generate_seeds(self, num_samples: int) -> List[SeedSpec]:
        """
        Generate stratified seed specifications.

        Ensures balanced representation across:
        1. GL accounts (equal samples per account)
        2. Vendor variants
        3. Languages
        4. Text styles
        5. Price distribution (with IT threshold oversampling)
        """
        seeds = []

        # Get all base categories
        all_categories = list(PRICE_RANGES.keys())

        # Calculate samples per base category
        # Note: IT_HARDWARE maps to 2 GL accounts based on price threshold
        num_gl_accounts = len(GLAccount)
        base_samples_per_gl = num_samples // num_gl_accounts

        # Track samples per GL account
        gl_counts = Counter()

        seed_id = 0

        # Generate seeds with stratification
        while len(seeds) < num_samples:
            # Cycle through categories to ensure coverage
            for category in all_categories:
                if len(seeds) >= num_samples:
                    break

                # Sample attributes
                department = self._sample_department(category)
                vendor = self.rng.choice(VENDOR_VARIANTS)
                language = self.rng.choice(list(Language))
                text_style = self.rng.choice(list(TextStyle))
                price = self._sample_price(category)

                # Determine GL account based on accounting rules
                gl_account = get_gl_account(category, price)

                # Check if this GL account is under-represented
                # (for balanced distribution)
                current_min_count = min(gl_counts.values()) if gl_counts else 0
                if gl_counts[gl_account] > current_min_count + 1:
                    # Skip to maintain balance
                    continue

                seed = SeedSpec(
                    seed_id=seed_id,
                    base_category=category,
                    department=department,
                    vendor=vendor,
                    language=language,
                    text_style=text_style,
                    net_amount_eur=price,
                    gl_account=gl_account,
                )

                seeds.append(seed)
                gl_counts[gl_account] += 1
                seed_id += 1

        # Shuffle to avoid ordering bias
        self.rng.shuffle(seeds)

        # Re-assign sequential seed_ids after shuffle
        for idx, seed in enumerate(seeds):
            seed.seed_id = idx

        return seeds[:num_samples]

    def generate_counterfactual_pairs(
        self, num_pairs: int
    ) -> List[tuple[SeedSpec, SeedSpec]]:
        """
        Generate counterfactual pairs for robustness testing.

        Two types:
        1. Price flip: Identical IT item at 749 EUR vs 849 EUR (GL threshold)
        2. Department flip: Identical item with different departments
        """
        pairs = []
        seed_id_offset = 100000  # High offset to avoid collision with main seeds

        for i in range(num_pairs):
            flip_type = "price" if i % 2 == 0 else "department"

            if flip_type == "price":
                # Price flip around 800 EUR threshold
                category = "IT_HARDWARE"
                department = self._sample_department(category)
                vendor = self.rng.choice(VENDOR_VARIANTS)
                language = self.rng.choice(list(Language))
                text_style = self.rng.choice(list(TextStyle))

                # Below threshold
                price_low = self.np_rng.uniform(700.0, 795.0)
                gl_low = get_gl_account(category, price_low)

                seed_low = SeedSpec(
                    seed_id=seed_id_offset + i * 2,
                    base_category=category,
                    department=department,
                    vendor=vendor,
                    language=language,
                    text_style=text_style,
                    net_amount_eur=round(price_low, 2),
                    gl_account=gl_low,
                )

                # Above threshold (same item, different price)
                price_high = self.np_rng.uniform(805.0, 900.0)
                gl_high = get_gl_account(category, price_high)

                seed_high = SeedSpec(
                    seed_id=seed_id_offset + i * 2 + 1,
                    base_category=category,
                    department=department,
                    vendor=vendor,
                    language=language,
                    text_style=text_style,
                    net_amount_eur=round(price_high, 2),
                    gl_account=gl_high,
                )

                pairs.append((seed_low, seed_high))

            else:
                # Department flip (same item, different departments)
                category = self.rng.choice(["OFFICE_CONSUMABLES", "TECH_TRAINING"])
                vendor = self.rng.choice(VENDOR_VARIANTS)
                language = self.rng.choice(list(Language))
                text_style = self.rng.choice(list(TextStyle))
                price = self._sample_price(category)
                gl_account = get_gl_account(category, price)

                # Two different departments
                dept1 = Department.MARKETING
                dept2 = Department.IT_OPS

                seed1 = SeedSpec(
                    seed_id=seed_id_offset + i * 2,
                    base_category=category,
                    department=dept1,
                    vendor=vendor,
                    language=language,
                    text_style=text_style,
                    net_amount_eur=round(price, 2),
                    gl_account=gl_account,
                )

                seed2 = SeedSpec(
                    seed_id=seed_id_offset + i * 2 + 1,
                    base_category=category,
                    department=dept2,
                    vendor=vendor,
                    language=language,
                    text_style=text_style,
                    net_amount_eur=round(price, 2),
                    gl_account=gl_account,
                )

                pairs.append((seed1, seed2))

        return pairs


def print_distribution_summary(seeds: List[SeedSpec]) -> None:
    """Print summary statistics of seed distribution."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # GL account distribution
    gl_counts = Counter(seed.gl_account for seed in seeds)
    table = Table(title="GL Account Distribution", show_header=True)
    table.add_column("GL Account", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")

    total = len(seeds)
    for gl_account in sorted(gl_counts.keys(), key=lambda x: x.value):
        count = gl_counts[gl_account]
        pct = (count / total) * 100
        table.add_row(gl_account.value, str(count), f"{pct:.1f}%")

    console.print(table)

    # IT hardware price distribution around threshold
    it_seeds = [s for s in seeds if s.base_category == "IT_HARDWARE"]
    if it_seeds:
        below_800 = sum(1 for s in it_seeds if s.net_amount_eur < 800)
        around_threshold = sum(1 for s in it_seeds if 650 <= s.net_amount_eur <= 950)

        console.print("\n[bold]IT Hardware Price Distribution:[/bold]")
        console.print(f"  Total IT hardware seeds: {len(it_seeds)}")
        console.print(f"  Below 800 EUR (GL_6220): {below_800}")
        console.print(f"  >= 800 EUR (GL_1400): {len(it_seeds) - below_800}")
        console.print(
            f"  In threshold region (650-950 EUR): {around_threshold} "
            f"({(around_threshold/len(it_seeds)*100):.1f}%)"
        )

    # Category distribution
    category_counts = Counter(seed.base_category for seed in seeds)
    table2 = Table(title="\nBase Category Distribution", show_header=True)
    table2.add_column("Category", style="cyan")
    table2.add_column("Count", justify="right", style="green")

    for category in sorted(category_counts.keys()):
        table2.add_row(category, str(category_counts[category]))

    console.print(table2)

    # Department distribution
    dept_counts = Counter(seed.department for seed in seeds)
    console.print(f"\n[bold]Department Distribution:[/bold]")
    for dept in sorted(dept_counts.keys(), key=lambda x: x.value):
        console.print(f"  {dept.value}: {dept_counts[dept]}")

    # Language distribution
    lang_counts = Counter(seed.language for seed in seeds)
    console.print(f"\n[bold]Language Distribution:[/bold]")
    for lang in sorted(lang_counts.keys(), key=lambda x: x.value):
        console.print(f"  {lang.value}: {lang_counts[lang]}")

    # Text style distribution
    style_counts = Counter(seed.text_style for seed in seeds)
    console.print(f"\n[bold]Text Style Distribution:[/bold]")
    for style in sorted(style_counts.keys(), key=lambda x: x.value):
        console.print(f"  {style.value}: {style_counts[style]}")


def main():
    """CLI entry point for testing seed generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate seed specifications")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DATA_GEN_CONFIG["test_samples"],
        help="Number of seeds to generate (default: 160 for testing)",
    )
    parser.add_argument(
        "--counterfactual-pairs",
        type=int,
        default=20,
        help="Number of counterfactual pairs to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path (optional)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DATA_GEN_CONFIG["random_seed"],
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Generate seeds
    generator = SeedGenerator(seed=args.seed)

    print(f"\n🌱 Generating {args.num_samples} seed specifications...\n")
    seeds = generator.generate_seeds(args.num_samples)

    # Print distribution
    print_distribution_summary(seeds)

    # Generate counterfactual pairs
    print(f"\n🔄 Generating {args.counterfactual_pairs} counterfactual pairs...\n")
    pairs = generator.generate_counterfactual_pairs(args.counterfactual_pairs)

    print(f"[bold green]✓ Generated {len(pairs)} counterfactual pairs[/bold green]")
    print("\nExample price flip pair:")
    pair = pairs[0]
    print(f"  Low:  {pair[0].net_amount_eur} EUR -> {pair[0].gl_account.value}")
    print(f"  High: {pair[1].net_amount_eur} EUR -> {pair[1].gl_account.value}")

    # Save to file if requested
    if args.output:
        import json

        output_data = {
            "seeds": [s.to_dict() for s in seeds],
            "counterfactual_pairs": [
                {"low": p[0].to_dict(), "high": p[1].to_dict()} for p in pairs
            ],
            "metadata": {
                "num_samples": len(seeds),
                "num_counterfactual_pairs": len(pairs),
                "random_seed": args.seed,
            },
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Saved to {args.output}")


if __name__ == "__main__":
    main()
