"""
Critic filter using zero-shot LLM verification.

Why we need this: the verbalizer sometimes generates descriptions that don't match
the seed category, or writes something ambiguous. This filter catches those before
they pollute the training data.

How it works: format each record as an ERP invoice line, ask an LLM to classify it
zero-shot (no examples, just the rules), and drop rows where the LLM's prediction
disagrees with the seed label. If a human accountant looking at the invoice line
wouldn't pick the same GL account we assigned, we don't want it in training.

Typical pass rate: 85-90%. Most rejections are legitimately ambiguous records.
"""

import json
from typing import List, Tuple

from pydantic import BaseModel, Field
from rich.console import Console
from tqdm import tqdm

from .config import GLAccount
from .llm_client import create_llm_client, BaseLLMClient

console = Console()


# GL account codes listed explicitly so the model can copy one verbatim
GL_ACCOUNTS = [
    "GL_6220_IT_PERIPHERALS_OPEX",
    "GL_1400_IT_HARDWARE_CAPEX",
    "GL_6400_OFFICE_CONSUMABLES",
    "GL_6410_FACILITY_PANTRY",
    "GL_6150_MARKETING_PROMO",
    "GL_6310_WORKPLACE_SAFETY_ERGONOMICS",
    "GL_6250_TECH_BOOKS_TRAINING",
    "GL_6450_CLEANING_MAINTENANCE",
]

# Plain-text prompt — no JSON, just ask for one GL code on a single line.
# Small models handle this much more reliably than structured output.
ZERO_SHOT_PROMPT = """Classify this Amazon Business invoice line into exactly one GL account.

Rules:
- IT items under 800 EUR → GL_6220_IT_PERIPHERALS_OPEX
- IT items 800 EUR or above → GL_1400_IT_HARDWARE_CAPEX
- Office supplies → GL_6400_OFFICE_CONSUMABLES
- Coffee, snacks, drinks → GL_6410_FACILITY_PANTRY
- Branded merchandise, promo → GL_6150_MARKETING_PROMO
- Ergonomic/safety equipment → GL_6310_WORKPLACE_SAFETY_ERGONOMICS
- Books, training materials → GL_6250_TECH_BOOKS_TRAINING
- Cleaning supplies → GL_6450_CLEANING_MAINTENANCE

Invoice: {invoice_line}

Reply with only the GL account code, nothing else."""


def format_erp_line(record: dict) -> str:
    """Format record as ERP invoice line for critic."""
    return (
        f"[VENDOR] {record['vendor']} "
        f"[DEPT] {record['department']} "
        f"[NET_AMT] {record['net_amount_eur']:.2f} "
        f"[CURR] EUR "
        f"[DESC] {record['description']}"
    )


class Critic:
    """Zero-shot LLM critic for filtering inconsistent records."""

    def __init__(
        self,
        llm_client: BaseLLMClient = None,
        backend: str = None,
        **llm_kwargs
    ):
        """
        Initialize critic.

        Args:
            llm_client: Pre-configured LLM client (optional)
            backend: LLM backend (openai, openrouter, bedrock)
            **llm_kwargs: Arguments passed to create_llm_client
        """
        if llm_client is None:
            self.llm = create_llm_client(backend=backend, **llm_kwargs)
        else:
            self.llm = llm_client

    def verify_one(self, record: dict) -> dict:
        """
        Verify a single record with zero-shot classification.

        Asks for a plain GL account code on a single line — no JSON needed.
        """
        try:
            invoice_line = format_erp_line(record)
            prompt = ZERO_SHOT_PROMPT.format(invoice_line=invoice_line)

            raw = self.llm.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=32,  # GL code is at most ~40 chars
            )

            # Extract the GL code: take the first token that matches a known code
            predicted_gl = None
            for token in raw.upper().split():
                token = token.strip(".,;:\n")
                for gl in GL_ACCOUNTS:
                    if token == gl:
                        predicted_gl = gl
                        break
                if predicted_gl:
                    break

            # Fallback: substring match in case the model adds extra words
            if predicted_gl is None:
                raw_upper = raw.upper()
                for gl in GL_ACCOUNTS:
                    if gl in raw_upper:
                        predicted_gl = gl
                        break

            if predicted_gl is None:
                predicted_gl = "UNKNOWN"

            true_gl = record['gl_account']
            record['predicted_gl'] = predicted_gl
            record['matches_label'] = predicted_gl == true_gl
            record['critic_confidence'] = "high" if predicted_gl != "UNKNOWN" else "low"
            record['critic_reasoning'] = raw.strip()
            return record

        except Exception as e:
            console.print(f"[red]Error verifying record: {e}[/red]")
            record['predicted_gl'] = "ERROR"
            record['matches_label'] = False
            record['critic_confidence'] = "error"
            record['critic_reasoning'] = str(e)
            return record

    def verify_batch(
        self,
        records: List[dict],
        show_progress: bool = True,
        batch_size: int = 64,
    ) -> List[dict]:
        """
        Verify a batch of records using batched inference when available.
        """
        prompts = [
            ZERO_SHOT_PROMPT.format(invoice_line=format_erp_line(r))
            for r in records
        ]

        if hasattr(self.llm, "generate_batch"):
            all_raws = []
            batches = range(0, len(prompts), batch_size)
            iterator = tqdm(batches, desc="Verifying records") if show_progress else batches
            for i in iterator:
                chunk = prompts[i:i + batch_size]
                raws = self.llm.generate_batch(chunk, temperature=0.0, max_tokens=32, batch_size=batch_size)
                all_raws.extend(raws)
        else:
            iterator = tqdm(prompts, desc="Verifying records") if show_progress else prompts
            all_raws = [self.llm.generate(p, temperature=0.0, max_tokens=32) for p in iterator]

        results = []
        for record, raw in zip(records, all_raws):
            record = record.copy()
            predicted_gl = None
            raw_upper = raw.upper()
            for token in raw_upper.split():
                token = token.strip(".,;:\n")
                if token in GL_ACCOUNTS:
                    predicted_gl = token
                    break
            if predicted_gl is None:
                for gl in GL_ACCOUNTS:
                    if gl in raw_upper:
                        predicted_gl = gl
                        break
            if predicted_gl is None:
                predicted_gl = "UNKNOWN"

            record['predicted_gl'] = predicted_gl
            record['matches_label'] = predicted_gl == record['gl_account']
            record['critic_confidence'] = "high" if predicted_gl != "UNKNOWN" else "low"
            record['critic_reasoning'] = raw.strip()
            results.append(record)

        return results

    def filter_verified(self, records: List[dict]) -> Tuple[List[dict], List[dict]]:
        """
        Filter records to keep only those where critic prediction matches seed label.

        Returns:
            (passed_records, rejected_records)
        """
        passed = [r for r in records if r.get('matches_label', False)]
        rejected = [r for r in records if not r.get('matches_label', False)]
        return passed, rejected


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Filter records with zero-shot critic")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file from corruptor",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file with filtered records",
    )
    parser.add_argument(
        "--rejected-output",
        type=str,
        help="Optional output file for rejected records",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "openrouter", "bedrock"],
        help="LLM backend (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name/ID (backend-specific)",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="Number of rejected examples to show",
    )

    args = parser.parse_args()

    # Load records
    console.print(f"📂 Loading records from {args.input}")
    records = []
    with open(args.input, 'r') as f:
        for line in f:
            records.append(json.loads(line))

    console.print(f"   Loaded {len(records)} records")

    # Create LLM client
    llm_kwargs = {}
    if args.model:
        if args.backend == "bedrock":
            llm_kwargs["model_id"] = args.model
        else:  # openai, openrouter
            llm_kwargs["model"] = args.model

    # Verify
    console.print(f"\n🔍 Verifying records with {args.backend} backend...\n")
    critic = Critic(backend=args.backend, **llm_kwargs)

    verified_records = critic.verify_batch(records, show_progress=True)

    # Filter
    passed, rejected = critic.filter_verified(verified_records)

    console.print(f"\n[bold]Verification Results:[/bold]")
    console.print(f"  Passed: {len(passed)}/{len(records)} ({len(passed)/len(records)*100:.1f}%)")
    console.print(f"  Rejected: {len(rejected)}/{len(records)} ({len(rejected)/len(records)*100:.1f}%)")

    # Rejection breakdown by GL account
    from collections import Counter
    rejection_by_gl = Counter(r['gl_account'] for r in rejected)
    if rejection_by_gl:
        console.print("\n[bold]Rejections by GL Account:[/bold]")
        for gl, count in sorted(rejection_by_gl.items()):
            console.print(f"  {gl}: {count}")

    # Save passed records
    console.print(f"\n💾 Saving passed records to {args.output}")
    with open(args.output, 'w') as f:
        for record in passed:
            f.write(json.dumps(record) + '\n')

    # Optionally save rejected records
    if args.rejected_output:
        console.print(f"💾 Saving rejected records to {args.rejected_output}")
        with open(args.rejected_output, 'w') as f:
            for record in rejected:
                f.write(json.dumps(record) + '\n')

    # Show examples of rejected records
    if rejected:
        console.print(f"\n[bold cyan]Rejected Examples ({min(args.show_examples, len(rejected))} samples):[/bold cyan]\n")
        for i, record in enumerate(rejected[:args.show_examples], 1):
            console.print(f"[yellow]Rejected {i}:[/yellow]")
            console.print(f"  True GL: {record['gl_account']}")
            console.print(f"  Predicted GL: {record['predicted_gl']}")
            console.print(f"  Confidence: {record['critic_confidence']}")
            console.print(f"  Reasoning: {record['critic_reasoning']}")
            console.print(f"  Invoice Line: {format_erp_line(record)}")
            console.print()

    console.print(f"[bold green]✓ Filtered {len(passed)} verified records[/bold green]")


if __name__ == "__main__":
    main()
