"""
LLM-based verbalizer for generating realistic Amazon Business item descriptions.

Takes a seed (category, price, department, language, style) and generates a plausible
Amazon product title and description. The LLM doesn't see the GL account label, just
the category and price. This prevents it from writing "IT peripheral" or "capital
equipment" in the description, which would leak the answer the classifier is supposed
to learn.

The prompts include price hints (< 800 EUR vs >= 800 EUR for IT hardware) because
that's information a person writing a product listing would have, but they never
mention the accounting classification.
"""

import json
from typing import List
from dataclasses import asdict

from pydantic import BaseModel, Field
from rich.console import Console
from tqdm import tqdm

from .seed_generator import SeedSpec
from .config import Language, TextStyle
from .llm_client import create_llm_client, BaseLLMClient

console = Console()


import hashlib
import string

def _fake_asin(seed_id: int) -> str:
    """Generate a deterministic plausible ASIN from a seed id."""
    h = hashlib.md5(str(seed_id).encode()).hexdigest().upper()
    return "B0" + h[:8]


# Plain-text prompts — no JSON schema, no structured output.
# Small models like TinyLlama are unreliable at JSON; plain text works well.
# We extract the first non-empty line as the product title and the rest as description.
CATEGORY_PROMPTS = {
    "IT_HARDWARE": {
        "low":  "Write a short Amazon product listing title and one sentence description for an IT {style} item costing {price:.0f} EUR. Examples: keyboard, webcam, USB hub, headset, 24-inch monitor. Language: {language}. Output title on line 1, description on line 2.",
        "high": "Write a short Amazon product listing title and one sentence description for a premium IT {style} item costing {price:.0f} EUR. Examples: 27-inch 4K monitor, ultrawide display, docking station. Language: {language}. Output title on line 1, description on line 2.",
    },
    "OFFICE_CONSUMABLES":    "Write a short Amazon product listing title and one sentence description for a business office consumable {style} item costing {price:.0f} EUR. Examples: HP toner cartridges, A4 copy paper (500 sheets), Leitz arch lever files, Pilot V5 pens, Stabilo highlighters, Post-it notes, Scotch tape, staples, paperclips, binder clips, whiteboard markers, correction fluid, label tape for DYMO printers. Language: {language}. Output title on line 1, description on line 2.",
    "FACILITY_PANTRY":       "Write a short Amazon product listing title and one sentence description for an office pantry {style} item costing {price:.0f} EUR. Examples: coffee pods, tea, snacks, mineral water, energy drinks. Language: {language}. Output title on line 1, description on line 2.",
    "MARKETING_PROMO":       "Write a short Amazon product listing title and one sentence description for a promotional merchandise {style} item costing {price:.0f} EUR. Examples: branded tote bag, custom USB drive, logoed water bottle, lanyards. Language: {language}. Output title on line 1, description on line 2.",
    "WORKPLACE_SAFETY":      "Write a short Amazon product listing title and one sentence description for a workplace ergonomics {style} item costing {price:.0f} EUR. Examples: monitor arm, anti-fatigue mat, ergonomic chair, wrist rest, footrest. Language: {language}. Output title on line 1, description on line 2.",
    "TECH_TRAINING":         "Write a short Amazon product listing title and one sentence description for a technical training {style} item costing {price:.0f} EUR. Examples: Python programming book, AWS certification guide, O'Reilly book, Linux manual. Language: {language}. Output title on line 1, description on line 2.",
    "CLEANING_MAINTENANCE":  "Write a short Amazon product listing title and one sentence description for a cleaning supply {style} item costing {price:.0f} EUR. Examples: disinfectant wipes, paper towels, trash bags, cleaning spray, microfibre cloths. Language: {language}. Output title on line 1, description on line 2.",
}

STYLE_LABELS = {
    TextStyle.FULL_MARKETPLACE_TITLE: "with full brand and spec details",
    TextStyle.SAP_SGTXT_ABBREVIATED:  "with abbreviated name only",
    TextStyle.ASIN_SKU_HEAVY:         "with product code or SKU",
    TextStyle.GENERIC_AMBIGUOUS:      "with a generic name only",
}


class Verbalizer:
    """Generates realistic Amazon item descriptions from seed specifications."""

    def __init__(
        self,
        llm_client: BaseLLMClient = None,
        backend: str = None,
        **llm_kwargs
    ):
        """
        Initialize verbalizer.

        Args:
            llm_client: Pre-configured LLM client (optional)
            backend: LLM backend (openai, openrouter, bedrock)
            **llm_kwargs: Arguments passed to create_llm_client
        """
        if llm_client is None:
            self.llm = create_llm_client(backend=backend, **llm_kwargs)
        else:
            self.llm = llm_client

    def _build_prompt(self, seed: SeedSpec) -> str:
        """Build a plain-text prompt — no JSON, no schema, just ask for two lines."""
        style_label = STYLE_LABELS.get(seed.text_style, "")
        template = CATEGORY_PROMPTS.get(seed.base_category, CATEGORY_PROMPTS["OFFICE_CONSUMABLES"])

        # IT_HARDWARE has two sub-templates based on price threshold
        if isinstance(template, dict):
            template = template["high"] if seed.net_amount_eur >= 800 else template["low"]

        return template.format(
            price=seed.net_amount_eur,
            style=style_label,
            language=seed.language.value,
        )

    @staticmethod
    def _parse_plain_text(text: str, seed_id: int) -> tuple[str, str]:
        """
        Extract title and description from a two-line plain-text response.

        Falls back gracefully: if the model only outputs one line we use it
        for both fields. If it outputs nothing we return placeholder strings.
        """
        # Strip any preamble like "Title:" or "Line 1:"
        lines = [
            l.strip()
            for l in text.strip().splitlines()
            if l.strip() and not l.strip().lower().startswith(("title:", "description:", "line "))
        ]

        if len(lines) >= 2:
            return lines[0], lines[1]
        elif len(lines) == 1:
            return lines[0], lines[0]
        else:
            return f"Amazon product {seed_id}", "Amazon Business item"

    def verbalize_one(self, seed: SeedSpec) -> dict:
        """Generate Amazon item description for a single seed."""
        try:
            prompt = self._build_prompt(seed)

            system_prompt = (
                "You write realistic Amazon product listings. "
                "Output exactly two lines: line 1 is the product title, line 2 is a one-sentence description. "
                "Never mention accounting codes, GL accounts, or expense categories."
            )

            # Plain text generation — no JSON parsing needed
            raw = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.8,
                max_tokens=128,
            )

            title, description = self._parse_plain_text(raw, seed.seed_id)
            asin = _fake_asin(seed.seed_id)

            result = asdict(seed)
            result.update({"title": title, "description": description, "asin": asin})
            return result

        except Exception as e:
            console.print(f"[red]Error verbalizing seed {seed.seed_id}: {e}[/red]")
            result = asdict(seed)
            result.update({"title": "Amazon product", "description": "Amazon Business item", "asin": _fake_asin(seed.seed_id)})
            return result

    def verbalize_batch(
        self,
        seeds: List[SeedSpec],
        show_progress: bool = True,
        batch_size: int = 32,
    ) -> List[dict]:
        """
        Generate descriptions for a batch of seeds.

        Uses batched inference when the backend supports it,
        falling back to sequential generation for others.
        """
        # Build all prompts up front
        prompts = [self._build_prompt(seed) for seed in seeds]
        system_prompt = (
            "You write realistic Amazon product listings. "
            "Output exactly two lines: line 1 is the product title, line 2 is a one-sentence description. "
            "Never mention accounting codes, GL accounts, or expense categories."
        )

        # Backends with native batch support
        if hasattr(self.llm, "generate_batch"):
            all_raws = []
            batches = range(0, len(prompts), batch_size)
            iterator = tqdm(batches, desc="Verbalizing seeds") if show_progress else batches
            for i in iterator:
                chunk_prompts = prompts[i:i + batch_size]
                raw_texts = self.llm.generate_batch(
                    chunk_prompts,
                    system_prompt=system_prompt,
                    temperature=0.8,
                    max_tokens=128,
                    batch_size=batch_size,
                )
                all_raws.extend(raw_texts)
        else:
            # Sequential generation
            iterator = tqdm(prompts, desc="Verbalizing seeds") if show_progress else prompts
            all_raws = [
                self.llm.generate(p, system_prompt=system_prompt, temperature=0.8, max_tokens=128)
                for p in iterator
            ]

        # Parse results
        results = []
        for seed, raw in zip(seeds, all_raws):
            try:
                title, description = self._parse_plain_text(raw, seed.seed_id)
                asin = _fake_asin(seed.seed_id)
                result = asdict(seed)
                result.update({"title": title, "description": description, "asin": asin})
            except Exception:
                result = asdict(seed)
                result.update({"title": "Amazon product", "description": "Amazon Business item", "asin": _fake_asin(seed.seed_id)})
            results.append(result)

        return results


def format_erp_line(record: dict) -> str:
    """Format record as ERP invoice line."""
    return (
        f"[VENDOR] {record['vendor']} "
        f"[DEPT] {record['department']} "
        f"[NET_AMT] {record['net_amount_eur']:.2f} "
        f"[CURR] EUR "
        f"[DESC] {record['description']}"
    )


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Amazon item descriptions from seeds")
    parser.add_argument(
        "--input",
        type=str,
        help="Input JSON file with seeds (from seed_generator)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of seeds to generate if not using input file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file",
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
        help="Number of examples to print",
    )

    args = parser.parse_args()

    # Load or generate seeds
    if args.input:
        console.print(f"📂 Loading seeds from {args.input}")
        with open(args.input, 'r') as f:
            data = json.load(f)
        seeds = [SeedSpec(**s) for s in data['seeds']]
    elif args.num_samples:
        from .seed_generator import SeedGenerator
        from .config import DATA_GEN_CONFIG

        console.print(f"🌱 Generating {args.num_samples} seeds")
        generator = SeedGenerator(seed=DATA_GEN_CONFIG["random_seed"])
        seeds = generator.generate_seeds(args.num_samples)
    else:
        console.print("[red]Error: Must provide --input or --num-samples[/red]")
        return

    console.print(f"\n🤖 Verbalizing {len(seeds)} seeds with {args.backend} backend...\n")

    # Create LLM client
    llm_kwargs = {}
    if args.model:
        if args.backend == "bedrock":
            llm_kwargs["model_id"] = args.model
        else:  # openai, openrouter
            llm_kwargs["model"] = args.model

    # Verbalize
    verbalizer = Verbalizer(backend=args.backend, **llm_kwargs)
    results = verbalizer.verbalize_batch(seeds, show_progress=True)

    # Save to JSONL
    console.print(f"\n💾 Saving to {args.output}")
    with open(args.output, 'w') as f:
        for record in results:
            f.write(json.dumps(record) + '\n')

    # Show examples
    console.print(f"\n[bold cyan]Examples ({args.show_examples} samples):[/bold cyan]\n")
    for i, record in enumerate(results[:args.show_examples], 1):
        console.print(f"[yellow]Example {i}:[/yellow]")
        console.print(f"  Category: {record['base_category']}")
        console.print(f"  GL: {record['gl_account']}")
        console.print(f"  Price: {record['net_amount_eur']:.2f} EUR")
        console.print(f"  Title: {record['title']}")
        console.print(f"  ASIN: {record['asin']}")
        console.print(f"  ERP Line: {format_erp_line(record)}")
        console.print()

    console.print(f"[bold green]✓ Generated {len(results)} verbalized records[/bold green]")


if __name__ == "__main__":
    main()
