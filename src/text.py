"""Text hygiene helpers shared by dataset building, evaluation and baselines.

The LLM that generated the product descriptions saw the invoice amount and
sometimes wrote it into the text ("... for 678 EUR"). That is a direct leak of
the numeric feature. `scrub_record` removes those explicit quotes at build time
and records that the source contained one, so evaluation can still slice on it.

(The opt-in price band token at the bottom of this module is the one deliberate
exception: it re-adds a rule-derived band label for the two-stage model.)

It deliberately does not try to remove the *style* correlation with price
("premium", "ultrawide", ...): that is a property of the generator, not an
explicit field leak, and removing it requires regenerating the text.
"""

import bisect
import json
import re
from collections.abc import Sequence
from pathlib import Path

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


def scrub_record(record: dict) -> dict:
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


# --- Price band token (two-stage model) -------------------------------------
# Stage 1 labels the amount with the band it falls in, using edges taken from the
# accounting rules. Stage 2 is the normal classifier, which sees that label as a
# token at the start of the text. Because the edges come from the labelling rule,
# this gives the model the answer for the amount-sensitive classes: treat results
# as a with-prior upper bound, not as evidence the boundary was learned.

PRICE_BAND_FILE = "price_band.json"
_BAND_RE = re.compile(r"^\[PRICE_BAND\]\s*\d+\s*")


def price_band_index(amount: float, edges: Sequence[float]) -> int:
    """Number of band edges at or below `amount` (an amount equal to an edge is above it)."""
    # bisect_right: an amount equal to an edge is "at or above" it. This matches rules of
    # the form "< edge" / ">= edge"; price_band_edges() refuses edges where it would not.
    return bisect.bisect_right(edges, amount)


def set_price_band(text: str, amount: float, edges: Sequence[float]) -> str:
    """Return `text` with its price band token set for `amount` (replacing any existing one)."""
    body = _BAND_RE.sub("", text)
    return f"[PRICE_BAND] {price_band_index(amount, edges)} {body}"


def sweep_texts(text: str, amounts: Sequence[float], edges: Sequence[float] | None) -> list[str]:
    """The text to score at each swept amount: the band token follows the amount, else fixed text."""
    if edges is None:
        return [text] * len(amounts)
    return [set_price_band(text, a, edges) for a in amounts]


def load_price_band_edges(directory: Path) -> list[float] | None:
    """Band edges saved next to a dataset or checkpoint, or None for a single-stage model."""
    path = Path(directory) / PRICE_BAND_FILE
    if not path.exists():
        return None
    with open(path) as f:
        return [float(e) for e in json.load(f)["edges"]]
