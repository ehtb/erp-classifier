"""
ERP data corruptor for realistic data augmentation.

Applies deterministic corruption patterns seen in real ERP systems:
1. SAP SGTXT 40-character truncation
2. PO number prefixes (PO#4500...)
3. OCR character substitutions (O->0, I->1, etc.)
"""

import json
import random

from rich.console import Console

console = Console()


# OCR substitution patterns (character-level noise)
OCR_SUBSTITUTIONS = {
    'O': '0',
    'o': '0',
    'I': '1',
    'l': '1',
    'S': '5',
    's': '5',
    'Z': '2',
    'B': '8',
    'G': '6',
}


class ERPCorruptor:
    """Applies realistic ERP data corruption patterns."""

    def __init__(
        self,
        corruption_ratio: float = 0.3,
        sgtxt_truncation_prob: float = 0.4,
        po_prefix_prob: float = 0.3,
        ocr_error_prob: float = 0.3,
        seed: int = 42,
    ):
        """
        Initialize corruptor.

        Args:
            corruption_ratio: Fraction of records to corrupt
            sgtxt_truncation_prob: P(apply SGTXT truncation | record selected for corruption)
            po_prefix_prob: P(add PO prefix | record selected for corruption)
            ocr_error_prob: P(apply OCR errors | record selected for corruption)
            seed: Random seed
        """
        self.corruption_ratio = corruption_ratio
        self.sgtxt_truncation_prob = sgtxt_truncation_prob
        self.po_prefix_prob = po_prefix_prob
        self.ocr_error_prob = ocr_error_prob
        self.rng = random.Random(seed)

    def _truncate_sgtxt(self, description: str) -> str:
        """
        Apply SAP SGTXT 40-character truncation.

        Mimics how SAP ERP truncates long text in the SGTXT field.
        """
        if len(description) <= 40:
            return description

        # Truncate at 40 chars, sometimes with ellipsis
        if self.rng.random() < 0.7:
            return description[:37] + "..."
        else:
            return description[:40]

    def _add_po_prefix(self, description: str) -> str:
        """Add PO number prefix like 'PO#4500123 - <description>'."""
        po_number = self.rng.randint(4500000, 4599999)
        return f"PO#{po_number} - {description}"

    def _apply_ocr_errors(self, text: str, error_rate: float = 0.05) -> str:
        """
        Apply OCR character substitution errors.

        Args:
            text: Input text
            error_rate: Probability of substitution per eligible character
        """
        chars = list(text)
        for i, char in enumerate(chars):
            if char in OCR_SUBSTITUTIONS and self.rng.random() < error_rate:
                chars[i] = OCR_SUBSTITUTIONS[char]
        return ''.join(chars)

    def corrupt_record(self, record: dict) -> dict:
        """
        Apply corruption to a single record.

        Returns new dict with 'corrupted' field and potentially modified description.
        """
        # Decide if this record gets corrupted
        if self.rng.random() >= self.corruption_ratio:
            record['corrupted'] = False
            record['corruption_types'] = []
            return record

        description = record['description']
        corruption_types = []

        # Apply corruption patterns
        if self.rng.random() < self.sgtxt_truncation_prob:
            description = self._truncate_sgtxt(description)
            corruption_types.append('sgtxt_truncation')

        if self.rng.random() < self.po_prefix_prob:
            description = self._add_po_prefix(description)
            corruption_types.append('po_prefix')

        if self.rng.random() < self.ocr_error_prob:
            description = self._apply_ocr_errors(description)
            corruption_types.append('ocr_errors')

        # Update record
        record['description'] = description
        record['corrupted'] = True
        record['corruption_types'] = corruption_types

        return record

    def corrupt_batch(self, records: list[dict]) -> list[dict]:
        """Apply corruption to a batch of records."""
        corrupted = []
        for record in records:
            corrupted.append(self.corrupt_record(record.copy()))
        return corrupted


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Apply ERP corruption to verbalized data")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file from verbalizer",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file with corruption",
    )
    parser.add_argument(
        "--corruption-ratio",
        type=float,
        default=0.3,
        help="Fraction of records to corrupt (default: 0.3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="Number of corrupted examples to show",
    )

    args = parser.parse_args()

    # Load records
    console.print(f"📂 Loading records from {args.input}")
    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))

    console.print(f"   Loaded {len(records)} records")

    # Apply corruption
    console.print(f"\n🔧 Applying corruption (ratio={args.corruption_ratio})...")
    corruptor = ERPCorruptor(
        corruption_ratio=args.corruption_ratio,
        seed=args.seed,
    )

    corrupted_records = corruptor.corrupt_batch(records)

    # Statistics
    num_corrupted = sum(1 for r in corrupted_records if r.get('corrupted', False))
    console.print(f"   Corrupted {num_corrupted}/{len(corrupted_records)} records ({num_corrupted/len(corrupted_records)*100:.1f}%)")

    # Corruption type breakdown
    corruption_type_counts = {}
    for record in corrupted_records:
        if record.get('corrupted', False):
            for ctype in record['corruption_types']:
                corruption_type_counts[ctype] = corruption_type_counts.get(ctype, 0) + 1

    console.print("\n[bold]Corruption Type Distribution:[/bold]")
    for ctype, count in sorted(corruption_type_counts.items()):
        console.print(f"  {ctype}: {count}")

    # Save
    console.print(f"\n💾 Saving to {args.output}")
    with open(args.output, 'w') as f:
        for record in corrupted_records:
            f.write(json.dumps(record) + '\n')

    # Show examples
    console.print(f"\n[bold cyan]Corrupted Examples ({args.show_examples} samples):[/bold cyan]\n")
    corrupted_samples = [r for r in corrupted_records if r.get('corrupted', False)][:args.show_examples]

    for i, record in enumerate(corrupted_samples, 1):
        console.print(f"[yellow]Example {i}:[/yellow]")
        console.print(f"  GL: {record['gl_account']}")
        console.print(f"  Price: {record['net_amount_eur']:.2f} EUR")
        console.print(f"  Corruption: {', '.join(record['corruption_types'])}")
        console.print(f"  Description: {record['description']}")
        console.print()

    console.print(f"[bold green]✓ Corrupted {len(corrupted_records)} records[/bold green]")


if __name__ == "__main__":
    main()
