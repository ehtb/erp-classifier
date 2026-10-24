"""
Non-neural baselines on the same splits, to show what the multimodal model has to beat.

Also runs the amount sweep for the tabular and text-only baselines: a text-only
model sees the same text at every amount, so it cannot produce a single flip no
matter how accurate it is on the real items. That is the premise of the demo.

Usage:
    python scripts/baselines.py --data-dir output/multimodal \
        --output output/evaluation-multimodal/baselines.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.boundaries import boundaries, near_boundary, sweep_amounts  # noqa: E402
from src.config import NUM_LABELS  # noqa: E402
from src.metrics import (  # noqa: E402
    aggregate_sweep,
    cluster_bootstrap_ci,
    sweep_case_metrics,
    wilson_ci,
)
from src.text import scrub_prices  # noqa: E402


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def tab(recs):
    return np.array([[r['metadata']['net_amount_eur'], r['dept_id']] for r in recs])


def text(recs):
    return [r['text'] for r in recs]


def weights(recs):
    return np.array([float(r.get('sample_weight', 1.0)) for r in recs])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def report(name, pred, recs, y, probs=None):
    n, k = len(y), int((pred == y).sum())
    lo, hi = wilson_ci(k, n)
    groups = [
        r.get('metadata', {}).get('group_id')
        if r.get('metadata', {}).get('group_id') is not None else r['text']
        for r in recs
    ]
    clo, chi = cluster_bootstrap_ci((pred == y).astype(float), groups)
    band = [
        p == t for r, p, t in zip(recs, pred, y)
        if near_boundary(r['metadata'].get('base_category'), r['metadata']['net_amount_eur'])
    ]
    print(f"{name:<38} acc={k/n:.3f} [{lo:.3f},{hi:.3f}]  "
          f"group CI [{clo:.3f},{chi:.3f}]  near-boundary {sum(band)}/{len(band)}")
    return {
        'accuracy': k / n,
        'ci95': [lo, hi],
        'ci95_group_bootstrap': [clo, chi],
        'n': n,
        'n_groups': len(set(groups)),
        'near_boundary_correct': int(sum(band)),
        'near_boundary_count': len(band),
    }


def full_probs(model, X):
    """Predict probabilities in a fixed NUM_LABELS-wide column order."""
    p = model.predict_proba(X)
    out = np.zeros((p.shape[0], NUM_LABELS), dtype=float)
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = p[:, j]
    return out


def sweep_baseline(name, proba, recs):
    """Run the fixed-text amount sweep for a baseline's probability function."""
    cases = []
    for r in recs:
        cat = r['metadata'].get('base_category')
        for b in boundaries(cat):
            amounts = sweep_amounts(b.amount)
            texts = [r['text']] * len(amounts)
            depts = [r['dept_id']] * len(amounts)
            p = proba(texts, amounts, depts)
            lo, hi = _label_id(b.below), _label_id(b.above)
            p_hi = p[:, hi] / (p[:, hi] + p[:, lo] + 1e-9)
            want_hi = amounts >= b.amount
            strict = p.argmax(-1) == np.where(want_hi, hi, lo)
            case = sweep_case_metrics(amounts, p_hi, b.amount)
            case['strict_correct'] = int(strict.sum())
            case['boundary'] = b.amount
            cases.append(case)
    agg = aggregate_sweep(cases)
    print(f"  sweep {name}: one flip {agg['exactly_one_flip_fraction']:.0%}, "
          f"crossing {agg['median_crossing_rel_error']} "
          f"(coverage {agg['crossing_coverage']:.0%})")
    return agg


def _label_id(label):
    from src.config import LABEL2ID
    return LABEL2ID[label]


def run_suite(tr, te, *, use_sample_weights: bool = True, skip_sweep: bool = False):
    """Fit and evaluate all non-neural baselines on `(tr, te)`."""
    ytr, yte = np.array([r['label'] for r in tr]), np.array([r['label'] for r in te])
    Xtr_t, Xte_t = tab(tr), tab(te)
    sw = weights(tr) if use_sample_weights else None
    out = {
        'n_train': len(tr),
        'n_test': len(te),
        'sample_weights': use_sample_weights,
    }

    dummy = DummyClassifier().fit(Xtr_t, ytr)
    out['majority'] = report("majority class", dummy.predict(Xte_t), te, yte)

    gb = HistGradientBoostingClassifier(categorical_features=[1], random_state=0)
    gb.fit(Xtr_t, ytr, sample_weight=sw)
    out['amount+dept GBDT'] = report("amount + dept (GBDT)", gb.predict(Xte_t), te, yte)

    def gb_proba(texts, amounts, depts):
        p = gb.predict_proba(np.array([amounts, depts]).T)
        full = np.zeros((len(amounts), NUM_LABELS))
        for j, cls in enumerate(gb.classes_):
            full[:, int(cls)] = p[:, j]
        return full

    # Text-only, normal and price-stripped
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, analyzer='word')
    Ttr, Tte = tfidf.fit_transform(text(tr)), tfidf.transform(text(te))
    lr = LogisticRegression(max_iter=3000, C=10)
    if sw is None:
        lr.fit(Ttr, ytr)
    else:
        lr.fit(Ttr, ytr, sample_weight=sw)
    out['tfidf LR (text only)'] = report("TF-IDF + LR (text only)", lr.predict(Tte), te, yte)

    def text_proba(texts, amounts, depts):
        return full_probs(lr, tfidf.transform(texts))

    # Price-stripped variant: same model class, refit on scrubbed text so the
    # comparison is not confounded by seeing the price at training time.
    tfidf_ps = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, analyzer='word')
    tr_ps = [scrub_prices(t) for t in text(tr)]
    te_ps = [scrub_prices(t) for t in text(te)]
    Ttr_ps, Tte_ps = tfidf_ps.fit_transform(tr_ps), tfidf_ps.transform(te_ps)
    lr_ps = LogisticRegression(max_iter=3000, C=10)
    if sw is None:
        lr_ps.fit(Ttr_ps, ytr)
    else:
        lr_ps.fit(Ttr_ps, ytr, sample_weight=sw)
    out['tfidf LR (price-stripped)'] = report(
        "TF-IDF + LR (price stripped)", lr_ps.predict(Tte_ps), te, yte
    )

    def text_ps_proba(texts, amounts, depts):
        return full_probs(lr_ps, tfidf_ps.transform([scrub_prices(t) for t in texts]))

    # Text + tabular: quantile-binned amount and one-hot dept appended to TF-IDF
    kb = KBinsDiscretizer(n_bins=40, encode='onehot', strategy='quantile', subsample=None)
    oh = OneHotEncoder(handle_unknown='ignore')
    # Unweighted amount quantiles match the PLE's default unweighted bins; the
    # model's family weights apply to the loss, not to feature resolution.
    Ftr = hstack([Ttr, kb.fit_transform(Xtr_t[:, :1]), oh.fit_transform(Xtr_t[:, 1:])]).tocsr()
    Fte = hstack([Tte, kb.transform(Xte_t[:, :1]), oh.transform(Xte_t[:, 1:])]).tocsr()
    lr_tab = LogisticRegression(max_iter=3000, C=10)
    if sw is None:
        lr_tab.fit(Ftr, ytr)
    else:
        lr_tab.fit(Ftr, ytr, sample_weight=sw)
    out['tfidf+tabular LR'] = report("TF-IDF + amount bins + dept (LR)", lr_tab.predict(Fte), te, yte)

    def tab_proba(texts, amounts, depts):
        T = tfidf.transform(texts)
        A = np.array(amounts, dtype=float).reshape(-1, 1)
        D = np.array(depts).reshape(-1, 1)
        F = hstack([T, kb.transform(A), oh.transform(D)]).tocsr()
        return full_probs(lr_tab, F)

    if not skip_sweep:
        out['sweeps'] = {
            'amount+dept GBDT': sweep_baseline("amount+dept GBDT", gb_proba, te),
            'tfidf LR (text only)': sweep_baseline("text only", text_proba, te),
            'tfidf LR (price-stripped)': sweep_baseline("text price-stripped", text_ps_proba, te),
            'tfidf+tabular LR': sweep_baseline("text+tabular", tab_proba, te),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="output/multimodal")
    ap.add_argument("--output", default="output/evaluation-multimodal/baselines.json")
    ap.add_argument("--no-sample-weights", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    d = Path(args.data_dir)
    tr, te = load(d / "train_multimodal.jsonl"), load(d / "test_multimodal.jsonl")
    use_sw = not args.no_sample_weights
    out = {
        'data_dir': str(d),
        'train_sha256': sha256(d / "train_multimodal.jsonl"),
        'test_sha256': sha256(d / "test_multimodal.jsonl"),
        'n_test': len(te),
    }

    has_cf = any(r.get('metadata', {}).get('counterfactual', False) for r in tr)
    if has_cf:
        tr_orig = [
            {**r, 'sample_weight': 1.0}
            for r in tr if not r.get('metadata', {}).get('counterfactual', False)
        ]
        print(f"=== Without counterfactuals (n_train={len(tr_orig)}) ===")
        out['without_counterfactuals'] = run_suite(
            tr_orig, te, use_sample_weights=use_sw, skip_sweep=args.skip_sweep,
        )
        print(f"\n=== With counterfactuals (n_train={len(tr)}) ===")
        out['with_counterfactuals'] = run_suite(
            tr, te, use_sample_weights=use_sw, skip_sweep=args.skip_sweep,
        )
    else:
        print(f"=== Without counterfactuals (n_train={len(tr)}) ===")
        out['without_counterfactuals'] = run_suite(
            tr, te, use_sample_weights=use_sw, skip_sweep=args.skip_sweep,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
