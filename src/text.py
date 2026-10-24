"""Text hygiene helpers shared by dataset building, evaluation and baselines.

The LLM that generated the product descriptions saw the invoice amount and
sometimes wrote it into the text ("... for 678 EUR"). That is a direct leak of
the numeric feature. `scrub_record` removes those explicit quotes at build time
and records that the source contained one, so evaluation can still slice on it.

It deliberately does not try to remove the *style* correlation with price
("premium", "ultrawide", ...): that is a property of the generator, not an
explicit field leak, and removing it requires regenerating the text.
"""

import re
from typing import Dict

# Currency-marked amounts in either order, plus the generator's standalone
# two-decimal format ("799.00"). One-decimal values ("2.5 m cable") are left
# alone on purpose.
_PRICE_RE = re.compile(
    r"""
    (?:
        (?:€|\bEUR\b|\bEuro\b)\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?
      | \d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?\s*(?:€|\bEUR\b|\bEuro\b)
      | \b\d{1,4}[.,]\d{2}\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Remove the quote entirely rather than replacing it with a marker: a "[PRICE]"
# token would still tell the model that this description contained an amount.
PRICE_PLACEHOLDER = ""


def contains_price(text: str) -> bool:
    return bool(text and _PRICE_RE.search(text))


def scrub_prices(text: str) -> str:
    if not text:
        return text
    out = _PRICE_RE.sub(PRICE_PLACEHOLDER, text)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"\(\s*\)", "", out)
    return out.strip()


def scrub_record(record: Dict) -> Dict:
    """Return a copy with explicit price quotes removed from description/title.

    Sets `price_quoted` on the copy so downstream evaluation can report the
    source-quoted slice even though the text no longer contains the amount.
    """
    new = dict(record)
    description = new.get("description", "")
    new["price_quoted"] = contains_price(description) or bool(new.get("price_quoted"))
    new["description"] = scrub_prices(description)
    if new.get("title"):
        new["title"] = scrub_prices(new["title"])
    return new
