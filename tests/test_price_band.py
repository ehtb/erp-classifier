import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from src import config
from src.boundaries import boundaries, price_band_edges
from src.config import ACCOUNTING_RULES, AccountingRule, GLAccount, PRICE_RANGES
from src.multimodal_dataset import MultimodalDatasetBuilder
from src.text import (
    PRICE_BAND_FILE,
    load_price_band_edges,
    price_band_index,
    set_price_band,
    sweep_texts,
)


def test_every_edge_is_a_rule_boundary():
    rule_boundaries = {b.amount for c in PRICE_RANGES for b in boundaries(c)}
    assert set(price_band_edges()) == rule_boundaries


def test_current_rules_have_one_edge_at_800():
    edges = price_band_edges()
    assert len(edges) == 1 and abs(edges[0] - 800.0) < 1e-6


def test_band_index_matches_rule_side():
    edges = price_band_edges()
    assert price_band_index(799.99, edges) == 0
    assert price_band_index(edges[0], edges) == 1  # an amount equal to the edge is "at or above"
    assert price_band_index(2000.0, edges) == 1


def test_band_numbering_with_several_edges():
    edges = [500.0, 1000.0]
    assert [price_band_index(a, edges) for a in (10, 499.99, 500, 999.99, 1000, 5000)] == [0, 0, 1, 1, 2, 2]


def test_set_price_band_replaces_instead_of_stacking():
    edges = price_band_edges()
    text = set_price_band("[VENDOR] Amazon [DESC] monitor", 750.0, edges)
    assert text.startswith("[PRICE_BAND] 0 ")
    flipped = set_price_band(text, 850.0, edges)
    assert flipped == "[PRICE_BAND] 1 [VENDOR] Amazon [DESC] monitor"
    assert flipped.count("[PRICE_BAND]") == 1


def test_sweep_texts_follow_the_amount_only_for_two_stage_models():
    amounts = [700.0, 900.0]
    assert sweep_texts("t", amounts, None) == ["t", "t"]
    assert sweep_texts("t", amounts, [800.0]) == ["[PRICE_BAND] 0 t", "[PRICE_BAND] 1 t"]


def test_load_price_band_edges_missing_file_and_round_trip():
    with tempfile.TemporaryDirectory() as d:
        assert load_price_band_edges(Path(d)) is None
        (Path(d) / PRICE_BAND_FILE).write_text(json.dumps({"edges": [500.0, 1000]}))
        assert load_price_band_edges(Path(d)) == [500.0, 1000.0]


def test_strict_greater_than_rule_is_refused():
    strict = [
        AccountingRule(GLAccount.GL_6220_IT_PERIPHERALS_OPEX, "IT_HARDWARE", "<= 800"),
        AccountingRule(GLAccount.GL_1400_IT_HARDWARE_CAPEX, "IT_HARDWARE", "> 800"),
    ] + [r for r in ACCOUNTING_RULES if r.base_category != "IT_HARDWARE"]
    boundaries.cache_clear()
    price_band_edges.cache_clear()
    try:
        with mock.patch.object(config, "ACCOUNTING_RULES", strict):
            with pytest.raises(ValueError, match="upper side"):
                price_band_edges()
    finally:
        boundaries.cache_clear()
        price_band_edges.cache_clear()


def _record(amount, **extra):
    return {
        'vendor': 'Amazon', 'description': 'monitor', 'department': 'IT_OPS',
        'net_amount_eur': amount, 'gl_account': 'GL_6220_IT_PERIPHERALS_OPEX', **extra,
    }


def test_builder_adds_token_only_when_enabled():
    b = MultimodalDatasetBuilder()
    b.dept_vocab.fit(['IT_OPS'])
    assert '[PRICE_BAND]' not in b.extract_features(_record(750.0)).text
    b.price_band_edges = list(price_band_edges())
    assert b.extract_features(_record(750.0)).text.startswith('[PRICE_BAND] 0 ')
    assert b.extract_features(_record(850.0)).text.startswith('[PRICE_BAND] 1 ')


def _source_records(n=80):
    recs = []
    for i in range(n):
        it = i % 2 == 0
        amount = 100.0 + 25.0 * (i % 40) if it else 20.0 + i
        recs.append({
            'vendor': 'Amazon', 'title': f"unique product {i} alpha{i} beta{i}",
            'description': f"description number {i} gamma{i}", 'department': 'IT_OPS',
            'net_amount_eur': amount,
            'base_category': 'IT_HARDWARE' if it else 'OFFICE_CONSUMABLES',
            'gl_account': (
                ('GL_1400_IT_HARDWARE_CAPEX' if amount >= 800 else 'GL_6220_IT_PERIPHERALS_OPEX')
                if it else 'GL_6400_OFFICE_CONSUMABLES'
            ),
        })
    return recs


def test_build_splits_writes_edges_and_tokens_for_counterfactuals_then_cleans_up():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        MultimodalDatasetBuilder(random_seed=1).build_splits(
            _source_records(), out, counterfactuals_per_record=2, price_band=True,
        )
        assert load_price_band_edges(out) == list(price_band_edges())
        rows = [json.loads(line) for line in open(out / "train_multimodal.jsonl")]
        assert any(r['metadata']['counterfactual'] for r in rows)
        for r in rows:
            band = 1 if r['metadata']['net_amount_eur'] >= 800 else 0
            assert r['text'].startswith(f"[PRICE_BAND] {band} "), r['text']

        # A later single-stage build into the same directory removes the stale file.
        MultimodalDatasetBuilder(random_seed=1).build_splits(_source_records(), out)
        assert not (out / PRICE_BAND_FILE).exists()
        assert '[PRICE_BAND]' not in json.loads(open(out / "train_multimodal.jsonl").readline())['text']
