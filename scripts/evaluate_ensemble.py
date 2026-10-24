"""
Evaluate a seed ensemble: accuracy, risk-coverage, and whether seed disagreement
flags the sweep points the model gets wrong.

This is the measurement the review asked for before promising abstention: a
confidence cut-off does not necessarily catch a 799 EUR error, but seed
disagreement might.

Usage:
    python scripts/evaluate_ensemble.py \
        --model-dirs models/ablation/D_ple_quantile_cf/seed42 \
                     models/ablation/D_ple_quantile_cf/seed43 \
                     models/ablation/D_ple_quantile_cf/seed44 \
        --test-data output/multimodal/test_multimodal.jsonl \
        --output output/evaluation-multimodal/ensemble.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.boundaries import boundaries, sweep_amounts  # noqa: E402
from src.config import LABEL2ID  # noqa: E402
from src.metrics import DEFAULT_STEP_TOL_REL, risk_coverage, sweep_case_metrics, wilson_ci  # noqa: E402
from src.model import get_device, load_model, predict_probs  # noqa: E402
from src.multimodal_dataset import load_multimodal_dataset  # noqa: E402


def load_models(model_dirs, device):
    return [load_model(Path(model_dir), device) for model_dir in model_dirs]


def predict_ensemble(models, records, device):
    return np.stack([
        predict_probs(
            model, tokenizer, [r['text'] for r in records],
            [r['num_features'] for r in records],
            [r['dept_id'] for r in records], device,
        )
        for model, tokenizer, _, _ in models
    ])  # [M, N, C]


def agreement_curve(agreement, correct):
    out = []
    for t in sorted(set(np.round(agreement, 2))):
        mask = agreement >= t
        out.append({
            'min_agreement': float(t),
            'coverage': float(mask.mean()),
            'accuracy': float(correct[mask].mean()),
            'n_accepted': int(mask.sum()),
        })
    return out


def sweep_ensemble(models, records, device):
    cases = []
    point_rows = []
    for r in records:
        cat = r['metadata'].get('base_category')
        for b in boundaries(cat):
            amounts = sweep_amounts(b.amount)
            per_model = []
            for model, tokenizer, mean, scale in models:
                feats = np.stack([amounts, np.log1p(amounts)], axis=1)
                z = ((feats - mean) / scale).astype(np.float32)
                p = predict_probs(
                    model, tokenizer, [r['text']] * len(amounts), z,
                    [r['dept_id']] * len(amounts), device,
                )
                lo, hi = LABEL2ID[b.below], LABEL2ID[b.above]
                per_model.append(p[:, hi] / (p[:, hi] + p[:, lo] + 1e-9))
            p_hi = np.stack(per_model)  # [M, A]
            mean_hi = p_hi.mean(0)
            std_hi = p_hi.std(0)
            case = sweep_case_metrics(amounts, mean_hi, b.amount)
            case['boundary'] = b.amount
            band = np.abs(amounts - b.amount) <= DEFAULT_STEP_TOL_REL * b.amount
            case['max_std_in_band'] = float(std_hi[band].max()) if band.any() else None
            cases.append(case)

            voters = p_hi >= 0.5
            agree = (voters == (mean_hi >= 0.5)).mean(0)
            correct = (mean_hi >= 0.5) == (amounts >= b.amount)
            for a, ag, ok, std in zip(amounts, agree, correct, std_hi):
                point_rows.append({
                    'amount': float(a), 'boundary': b.amount,
                    'agreement': float(ag), 'correct': bool(ok), 'std': float(std),
                    'in_band': bool(abs(a - b.amount) <= DEFAULT_STEP_TOL_REL * b.amount),
                })

    paired = [
        (abs(c['crossing'] - c['boundary']) / c['boundary'], c['max_std_in_band'])
        for c in cases
        if c['crossing'] is not None and c['max_std_in_band'] is not None
    ]
    errors = [e for e, _ in paired]
    stds = [s for _, s in paired]
    corr = None
    if len(paired) > 2 and np.std(errors) > 0 and np.std(stds) > 0:
        corr = float(np.corrcoef(errors, stds)[0, 1])

    def summarize(rows):
        if not rows:
            return None
        return {
            'n': len(rows),
            'error_rate': float(np.mean([not x['correct'] for x in rows])),
            'mean_agreement': float(np.mean([x['agreement'] for x in rows])),
        }

    in_band = [x for x in point_rows if x['in_band']]
    return {
        'cases': len(cases),
        'one_flip_fraction': float(np.mean([c['one_flip'] for c in cases])) if cases else None,
        'median_crossing_rel_error': float(np.median(errors)) if errors else None,
        'crossing_coverage': len(errors) / len(cases) if cases else None,
        'corr_disagreement_vs_crossing_error': corr,
        'in_band': summarize(in_band),
        'accepted_all_seeds_agree_in_band': summarize(
            [x for x in in_band if x['agreement'] >= 1.0]),
        'rejected_disagreement_in_band': summarize(
            [x for x in in_band if x['agreement'] < 1.0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dirs', nargs='+', required=True)
    ap.add_argument('--test-data', required=True)
    ap.add_argument('--output', default='output/evaluation-multimodal/ensemble.json')
    args = ap.parse_args()

    device = get_device()
    test = load_multimodal_dataset(Path(args.test_data))

    models = load_models(args.model_dirs, device)
    test_probs = predict_ensemble(models, test, device)
    mean_probs = test_probs.mean(0)
    preds = mean_probs.argmax(-1)
    labels = np.array([r['label'] for r in test])
    correct = preds == labels
    confidence = mean_probs.max(-1)
    agreement = (test_probs.argmax(-1) == preds).mean(0)

    k, n = int(correct.sum()), len(labels)
    curve, chosen = risk_coverage(confidence, correct.astype(float))
    results = {
        'model_dirs': args.model_dirs,
        'n_models': len(args.model_dirs),
        'n_test': n,
        'accuracy': k / n,
        'accuracy_ci95': wilson_ci(k, n),
        'mean_seed_agreement': float(agreement.mean()),
        'risk_coverage': curve,
        'cutoff_for_99pct': chosen,
        'agreement_coverage': agreement_curve(agreement, correct.astype(float)),
        'sweep': sweep_ensemble(models, test, device),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != 'risk_coverage'}, indent=2))
    print(f"Saved {args.output}")


if __name__ == '__main__':
    main()
