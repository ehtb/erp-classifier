"""
LLM-based verbalizer for generating realistic Amazon Business item descriptions.

Takes a seed (category, price, department, language, style) and generates a plausible
Amazon product title and description. The LLM doesn't see the GL account label, just
the category and price. This prevents it from writing "IT peripheral" or "capital
equipment" in the description, which would leak the answer the classifier is supposed
to learn.

The prompts include price hints (below vs at/above the IT hardware rule threshold)
because that's information a person writing a product listing would have, but they
never mention the accounting classification. The threshold is discovered from
``src/boundaries.py``, not hard-coded here.
"""

import hashlib
import json
import re
from dataclasses import asdict

from rich.console import Console
from tqdm import tqdm

from .boundaries import boundaries
from .config import TextStyle
from .llm_client import BaseLLMClient, create_llm_client
from .seed_generator import SeedSpec

console = Console()

SYSTEM_PROMPT = (
    "You write realistic Amazon product listings. "
    "Output exactly two lines: line 1 is the product title, line 2 is a one-sentence description. "
    "Never mention accounting codes, GL accounts, or expense categories."
)

_PREFIX_RE = re.compile(
    r"^\s*(?:title|description|line\s*\d+)\s*[:.\-]\s*", re.IGNORECASE
)


def _it_threshold() -> float:
    """IT hardware cutoff, discovered from the accounting rules."""
    found = boundaries("IT_HARDWARE")
    if not found:
        raise ValueError("IT_HARDWARE has no amount boundary in the accounting rules")
    return found[0].amount


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

        # IT_HARDWARE has two sub-templates based on the rule threshold
        if isinstance(template, dict):
            template = (
                template["high"]
                if seed.net_amount_eur >= _it_threshold()
                else template["low"]
            )

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
        Prefixes such as ``Title:`` and ``Line 1:`` are stripped, not dropped.
        """
        lines = []
        for raw_line in (text or "").splitlines():
            line = _PREFIX_RE.sub("", raw_line).strip()
            if line:
                lines.append(line)

        if len(lines) >= 2:
            return lines[0], lines[1]
        elif len(lines) == 1:
            return lines[0], lines[0]
        else:
            return f"Amazon product {seed_id}", "Amazon Business item"

    @staticmethod
    def _fallback_result(seed: SeedSpec, error: object | None = None) -> dict:
        """Placeholder used when generation fails; records the error for filtering."""
        result = asdict(seed)
        result.update(
            {
                "title": "Amazon product",
                "description": "Amazon Business item",
                "asin": _fake_asin(seed.seed_id),
            }
        )
        if error is not None:
            result["generation_error"] = str(error)
        return result

    def _generate_raws(
        self,
        prompts: list[str],
        *,
        show_progress: bool = True,
        batch_size: int = 32,
    ) -> list[str | None]:
        """Generate raw text for prompts, falling back to per-row calls on batch errors."""
        all_raws: list[str | None] = []
        batches = range(0, len(prompts), batch_size)
        iterator = tqdm(batches, desc="Verbalizing seeds") if show_progress else batches

        for i in iterator:
            chunk = prompts[i:i + batch_size]
            try:
                raw_texts = self.llm.generate_batch(
                    chunk,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.8,
                    max_tokens=128,
                    batch_size=batch_size,
                )
            except Exception as exc:
                console.print(
                    f"[yellow]Batch generation failed ({exc}); retrying row by row.[/yellow]"
                )
                raw_texts = []
                for prompt in chunk:
                    try:
                        raw_texts.append(
                            self.llm.generate(
                                prompt,
                                system_prompt=SYSTEM_PROMPT,
                                temperature=0.8,
                                max_tokens=128,
                            )
                        )
                    except Exception as row_exc:
                        console.print(f"[red]Error verbalizing a seed: {row_exc}[/red]")
                        raw_texts.append(None)

            if len(raw_texts) < len(chunk):
                raw_texts = list(raw_texts) + [None] * (len(chunk) - len(raw_texts))
            all_raws.extend(raw_texts[: len(chunk)])
        return all_raws

    def verbalize_one(self, seed: SeedSpec) -> dict:
        """Generate Amazon item description for a single seed."""
        try:
            prompt = self._build_prompt(seed)

            # Plain text generation — no JSON parsing needed
            raw = self.llm.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.8,
                max_tokens=128,
            )
            if not raw or not raw.strip():
                raise ValueError("empty response from LLM")

            title, description = self._parse_plain_text(raw, seed.seed_id)
            asin = _fake_asin(seed.seed_id)

            result = asdict(seed)
            result.update({"title": title, "description": description, "asin": asin})
            return result

        except Exception as e:
            console.print(f"[red]Error verbalizing seed {seed.seed_id}: {e}[/red]")
            return self._fallback_result(seed, e)

    def verbalize_batch(
        self,
        seeds: list[SeedSpec],
        show_progress: bool = True,
        batch_size: int = 32,
    ) -> list[dict]:
        """
        Generate descriptions for a batch of seeds.

        A failed prompt or API call produces a placeholder result with a
        ``generation_error`` field instead of aborting the whole run. The
        critic can then reject those rows.
        """
        prompts: list[str | None] = []
        prompt_errors = {}
        for idx, seed in enumerate(seeds):
            try:
                prompts.append(self._build_prompt(seed))
            except Exception as exc:
                prompts.append(None)
                prompt_errors[idx] = exc

        valid_indices = [i for i, prompt in enumerate(prompts) if prompt is not None]
        valid_prompts = [prompts[i] for i in valid_indices]
        raw_by_index = dict(
            zip(
                valid_indices,
                self._generate_raws(
                    valid_prompts,
                    show_progress=show_progress,
                    batch_size=batch_size,
                ),
            )
        )

        results = []
        for idx, seed in enumerate(seeds):
            if idx in prompt_errors:
                results.append(self._fallback_result(seed, prompt_errors[idx]))
                continue
            raw = raw_by_index.get(idx)
            if not raw or not raw.strip():
                results.append(self._fallback_result(seed, "generation failed or empty response"))
                continue
            try:
                title, description = self._parse_plain_text(raw, seed.seed_id)
                result = asdict(seed)
                result.update(
                    {
                        "title": title,
                        "description": description,
                        "asin": _fake_asin(seed.seed_id),
                    }
                )
            except Exception as exc:
                result = self._fallback_result(seed, exc)
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
        with open(args.input) as f:
            data = json.load(f)
        seeds = [SeedSpec.from_dict(s) for s in data['seeds']]
    elif args.num_samples:
        from .config import DATA_GEN_CONFIG
        from .seed_generator import SeedGenerator

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
