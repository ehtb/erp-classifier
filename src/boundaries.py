"""
Amount boundaries derived from the accounting rules in `config.py`.

The rules (`get_gl_account`) are the single source of truth for how the amount
decides the GL account. Everything here discovers the decision boundaries by
probing that function, so no threshold value appears anywhere in the model,
the dataset builder or the evaluation code. Change the rules and the
boundary-aware splits, counterfactual augmentation and sweep tests follow.
"""

import math
from functools import lru_cache
from typing import List, NamedTuple, Optional

import numpy as np

from .config import PRICE_RANGES, get_gl_account


class Boundary(NamedTuple):
    amount: float
    below: str  # GL account just below the boundary
    above: str  # GL account at/above the boundary


def _gl(category: str, amount: float) -> str:
    return get_gl_account(category, amount).value


@lru_cache(maxsize=None)
def boundaries(category: Optional[str]) -> List[Boundary]:
    """Amounts at which the GL account for `category` changes (empty if amount-insensitive)."""
    if category is None or category not in PRICE_RANGES:
        return []
    lo, hi = PRICE_RANGES[category]
    grid = np.geomspace(lo * 0.25, hi * 4, 4000)
    labels = [_gl(category, a) for a in grid]

    found = []
    for i in range(len(grid) - 1):
        if labels[i] != labels[i + 1]:
            a, b = grid[i], grid[i + 1]
            for _ in range(60):  # bisect to the exact switch point
                mid = (a + b) / 2
                if _gl(category, mid) == labels[i]:
                    a = mid
                else:
                    b = mid
            found.append(Boundary(round(float(b), 6), labels[i], labels[i + 1]))
    return found


def amount_sensitive(category: Optional[str]) -> bool:
    return bool(boundaries(category))


def near_boundary(category: Optional[str], amount: float, rel: float = 0.125) -> bool:
    """True when `amount` lies within +/- rel (relative) of any decision boundary."""
    return any(abs(amount - b.amount) <= rel * b.amount for b in boundaries(category))


def counterfactual_amounts(category: str, n: int, rng: np.random.Generator, rel: float = 0.3) -> List[float]:
    """
    Amounts log-uniformly spread within +/- rel of the category's boundaries,
    alternating below/above so both sides are equally represented.
    """
    bs = boundaries(category)
    out = []
    for i in range(n):
        b = bs[int(rng.integers(len(bs)))].amount
        if i % 2 == 0:
            lo, hi = b * (1 - rel), b * (1 - 1e-4)
        else:
            lo, hi = b, b * (1 + rel)
        out.append(round(float(math.exp(rng.uniform(math.log(lo), math.log(hi)))), 2))
    return out


def sweep_amounts(boundary: float) -> np.ndarray:
    """Dense log-spaced grid around a boundary plus exact near-miss offsets."""
    grid = np.geomspace(boundary * 0.6, boundary * 1.5, 61)
    rel_offsets = np.array([-0.0125, -0.00625, -0.00125, -1e-5, 0.0, 1e-5, 0.00125, 0.00625, 0.0125])
    return np.unique(np.round(np.concatenate([grid, boundary * (1 + rel_offsets)]), 2))
