import numpy as np
import pytest

from src.boundaries import amount_sensitive, boundaries, counterfactual_amounts, near_boundary
from src.config import AccountingRule, GLAccount, get_gl_account
from src.metrics import (
    DEFAULT_STEP_TOL_REL,
    aggregate_sweep,
    cross_threshold,
    expected_calibration_error,
    risk_coverage,
    sweep_case_metrics,
)
from src.model import PiecewiseLinearEncoding
from src.multimodal_dataset import (
    DepartmentVocabulary,
    MultimodalDatasetBuilder,
    MultimodalFeatures,
    near_duplicate_groups,
)
from src.text import contains_price, scrub_prices


def test_scrub_removes_currency_and_standalone_amounts():
    assert "678" not in scrub_prices("docking station for 678 EUR, USB-C")
    assert "799.00" not in scrub_prices("[NET_AMT] 799.00 monitor")
    assert scrub_prices("€ 1.234,56 monitor").strip() == "monitor"


def test_scrub_does_not_match_kleur_or_model_numbers():
    assert scrub_prices("PP klemmap in de kleur blauw") == "PP klemmap in de kleur blauw"
    assert scrub_prices("Logitech MK235 keyboard") == "Logitech MK235 keyboard"
    assert not contains_price("Logitech MK235 keyboard")


def test_cross_threshold_is_interpolated_in_log_amount():
    amounts = np.array([700.0, 790.0, 800.0, 810.0])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    result = cross_threshold(amounts, p)
    assert not result["censored"]
    assert not result["starts_above"]
    # 0.5 lies between 790 (p=.2) and 800 (p=.8); log interpolation puts it near 795.
    assert 790 < result["crossing"] < 800


def test_all_opex_is_censored_and_all_capex_starts_above():
    amounts = np.array([700.0, 800.0, 900.0])
    assert cross_threshold(amounts, np.zeros(3))["censored"]
    result = cross_threshold(amounts, np.ones(3))
    assert result["starts_above"] and result["crossing"] == 700.0


def test_tolerance_step_excludes_near_boundary_points():
    amounts = np.array([700.0, 790.0, 799.0, 800.0, 801.0, 810.0, 900.0])
    # Flips at ~795: wrong at 799 (inside the 2% band), right outside it.
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.95])
    case = sweep_case_metrics(amounts, p, 800.0, step_tol_rel=0.02)
    assert case["one_flip"] == 1
    assert case["tolerance_step"] == 1
    assert case["tight_correct"] < case["tight_points"]


def test_aggregate_reports_coverage_not_just_median():
    amounts = np.array([700.0, 800.0, 900.0])
    cases = []
    for p in (np.array([0.1, 0.9, 0.95]), np.zeros(3)):
        case = sweep_case_metrics(amounts, p, 800.0)
        case["boundary"] = 800.0
        cases.append(case)
    agg = aggregate_sweep(cases)
    assert agg["cases"] == 2
    assert agg["crossing_coverage"] == 0.5
    assert agg["crossing_censored_fraction"] == 0.5
    assert agg["median_crossing_rel_error"] is not None
    assert agg["step_tolerance_rel"] == DEFAULT_STEP_TOL_REL == 0.02


def test_ple_requires_at_least_two_edges():
    with pytest.raises(ValueError, match="at least 2 bin edges"):
        PiecewiseLinearEncoding(None, feature_idx=1, input_dim=2)
    with pytest.raises(ValueError, match="at least 2 bin edges"):
        PiecewiseLinearEncoding([0.0], feature_idx=1, input_dim=2)


def test_fit_bin_edges_weighted_vs_unweighted():
    feats = [
        MultimodalFeatures("a", np.array([0.0, 0.0], dtype=np.float32), 1, 0, {}, sample_weight=1.0),
        MultimodalFeatures("b", np.array([1.0, 1.0], dtype=np.float32), 1, 0, {}, sample_weight=1.0),
    ] + [
        MultimodalFeatures("c", np.array([10.0, 10.0], dtype=np.float32), 1, 1, {}, sample_weight=0.2)
        for _ in range(5)
    ]
    unweighted = MultimodalDatasetBuilder.fit_bin_edges(feats, "quantile", n_bins=2, weighting="unweighted")
    weighted = MultimodalDatasetBuilder.fit_bin_edges(feats, "quantile", n_bins=2, weighting="weighted")
    # Unweighted median sits at 10.0 (5 of 7 items); weighted median sits near 1.0.
    assert unweighted == [0.0, 10.0]
    assert weighted[1] < 5.0


def test_accounting_rule_rejects_unparseable_threshold():
    with pytest.raises(ValueError, match="Invalid threshold_rule"):
        AccountingRule(
            gl_account=GLAccount.GL_6220_IT_PERIPHERALS_OPEX,
            base_category="IT_HARDWARE",
            threshold_rule="=< 800",
        )


def test_boundaries_discovers_800_eur_it_threshold():
    it_bounds = boundaries("IT_HARDWARE")
    assert len(it_bounds) == 1
    b = it_bounds[0]
    assert pytest.approx(b.amount, rel=1e-5) == 800.0
    assert b.below == GLAccount.GL_6220_IT_PERIPHERALS_OPEX.value
    assert b.above == GLAccount.GL_1400_IT_HARDWARE_CAPEX.value
    assert amount_sensitive("IT_HARDWARE")
    assert not amount_sensitive("OFFICE_CONSUMABLES")
    assert boundaries("OFFICE_CONSUMABLES") == []
    assert near_boundary("IT_HARDWARE", 750.0, rel=0.125)
    assert not near_boundary("IT_HARDWARE", 500.0, rel=0.125)

    rng = np.random.default_rng(42)
    cfs = counterfactual_amounts("IT_HARDWARE", 4, rng)
    assert len(cfs) == 4
    # Even indices are < 800 (OPEX), odd indices are >= 800 (CAPEX)
    assert cfs[0] < 800.0 and cfs[2] < 800.0
    assert cfs[1] >= 800.0 and cfs[3] >= 800.0
    assert get_gl_account("IT_HARDWARE", cfs[0]) == GLAccount.GL_6220_IT_PERIPHERALS_OPEX
    assert get_gl_account("IT_HARDWARE", cfs[1]) == GLAccount.GL_1400_IT_HARDWARE_CAPEX


def test_group_split_is_deterministic_and_keeps_groups_intact():
    records = []
    for fam in range(12):
        cat = "IT_HARDWARE" if fam % 2 == 0 else "OFFICE_CONSUMABLES"
        for variant in range(2):
            records.append({
                "title": f"BrandX Family{fam} Pro Model {fam * 100} Variant {variant}",
                "description": f"BrandX Family{fam} Pro Model {fam * 100}",
                "gl_account": "GL_6220_IT_PERIPHERALS_OPEX" if cat == "IT_HARDWARE" else "GL_6400_OFFICE_CONSUMABLES",
                "base_category": cat,
                "net_amount_eur": 400.0,
            })
    groups = np.array(near_duplicate_groups(records))
    strata = np.array([r["gl_account"] for r in records])

    # Every 2-item product family must share one group ID
    for fam in range(12):
        fam_groups = set(groups[fam * 2:(fam + 1) * 2])
        assert len(fam_groups) == 1

    tr1, va1, te1 = MultimodalDatasetBuilder._greedy_group_split(records, groups, strata, 0.7, 0.15)
    tr2, va2, te2 = MultimodalDatasetBuilder._greedy_group_split(records, groups, strata, 0.7, 0.15)
    assert np.array_equal(tr1, tr2)
    assert np.array_equal(va1, va2)
    assert np.array_equal(te1, te2)

    # Disjoint indices covering all records, and zero group overlap across splits
    tr_g, va_g, te_g = set(groups[tr1]), set(groups[va1]), set(groups[te1])
    assert tr_g.isdisjoint(va_g)
    assert tr_g.isdisjoint(te_g)
    assert va_g.isdisjoint(te_g)
    assert len(tr1) + len(va1) + len(te1) == len(records)


def test_near_duplicate_groups_handles_repeated_generic_titles():
    records = [
        {"title": "Dell monitor", "description": "Dell monitor"}
        for _ in range(100)
    ]
    groups = near_duplicate_groups(records)
    assert len(set(groups)) == 1


def test_department_vocabulary_fit_resets_existing_ids():
    vocab = DepartmentVocabulary()
    vocab.fit(["IT_OPS"])
    vocab.fit(["MARKETING"])

    assert vocab.dept2id == {"UNK": 0, "MARKETING": 1}
    assert vocab.id2dept == {0: "UNK", 1: "MARKETING"}


def test_ece_and_risk_coverage_helpers():
    conf = np.array([0.55, 0.75, 0.92, 0.98])
    correct = np.array([0.0, 1.0, 1.0, 1.0])
    ece = expected_calibration_error(conf, correct)
    assert 0.0 <= ece <= 1.0
    curve, chosen = risk_coverage(conf, correct, target=0.99)
    assert chosen is not None
    assert chosen["cutoff"] == 0.60
    assert chosen["accuracy"] == 1.0
    assert chosen["n_accepted"] == 3
