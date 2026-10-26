"""
Evaluate the multimodal ERP classifier.

Reports, beyond plain accuracy:
  - Wilson 95% CIs and expected calibration error (ECE)
  - Accuracy on real test items near a decision boundary of the accounting rules
  - Amount sweep: every amount-sensitive test description re-scored around each
    rule boundary with text and department held fixed
  - Confidence cut-off selected on the validation set, then reported once on test
  - Boundary-crossing curve plot (`sweep_curve.png`, or `boundary_sweep.png` with `--compare-model-dir`)

Boundaries are discovered from the accounting rules (src/boundaries.py); nothing
about a specific threshold is hard-coded here.

Usage:
    python scripts/evaluate.py \
        --model-dir models/multimodal-classifier \
        --val-data output/multimodal/val_multimodal.jsonl \
        --test-data output/multimodal/test_multimodal.jsonl \
        --output-dir output/evaluation-multimodal
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.boundaries import amount_sensitive, boundaries, near_boundary, sweep_amounts  # noqa: E402
from src.config import ID2LABEL, LABEL2ID  # noqa: E402
from src.metrics import (  # noqa: E402
    DEFAULT_STEP_TOL_REL,
    aggregate_sweep,
    cluster_bootstrap_ci,
    expected_calibration_error,
    risk_coverage,
    sweep_case_metrics,
    wilson_ci,
)
from src.model import get_device, load_model, predict_probs  # noqa: E402
from src.multimodal_dataset import load_multimodal_dataset  # noqa: E402
from src.text import load_price_band_edges, scrub_prices, sweep_texts  # noqa: E402

console = Console()

NEAR_REL = 0.125   # "near a boundary" = within +/-12.5% of it
TIGHT_REL = 0.0625


def run_split(model, tokenizer, records, device):
    probs = predict_probs(
        model, tokenizer,
        [r['text'] for r in records],
        [r['num_features'] for r in records],
        [r['dept_id'] for r in records],
        device,
    )
    labels = np.array([r['label'] for r in records])
    preds = probs.argmax(-1)
    return probs, preds, labels


def boundary_analysis(records, preds, labels):
    """Accuracy on amount-sensitive items, and on those close to a rule boundary (real amounts)."""
    rows = [
        (r['metadata']['base_category'], r['metadata']['net_amount_eur'], int(p == t))
        for r, p, t in zip(records, preds, labels) if amount_sensitive(r['metadata'].get('base_category'))
    ]

    def acc(sel):
        k, n = sum(c for *_, c in sel), len(sel)
        return {'correct': k, 'count': n, 'accuracy': k / n if n else None, 'ci95': wilson_ci(k, n)}

    def dist(cat, amount):
        return min(abs(amount - b.amount) / b.amount for b in boundaries(cat))

    return {
        'amount_sensitive_items': acc(rows),
        'near_boundary_12.5pct': acc([x for x in rows if dist(x[0], x[1]) <= NEAR_REL]),
        'near_boundary_6.25pct': acc([x for x in rows if dist(x[0], x[1]) <= TIGHT_REL]),
        'closest_items': sorted(
            [{'category': c, 'amount': a, 'correct': bool(ok), 'rel_distance': dist(c, a)} for c, a, ok in rows],
            key=lambda d: d['rel_distance'],
        )[:8],
    }


def record_groups(records):
    """Group id per test record, falling back to title near-duplicate grouping."""
    ids = [r.get('metadata', {}).get('group_id') for r in records]
    if all(x is not None for x in ids):
        return ids
    from src.multimodal_dataset import near_duplicate_groups
    return near_duplicate_groups([{'title': r['text']} for r in records])


def evaluate_text_variant(model, tokenizer, records, texts, device):
    """Accuracy/IT/near-boundary accuracy for one text variant (price strip, constant)."""
    probs = predict_probs(
        model, tokenizer, texts,
        [r['num_features'] for r in records],
        [r['dept_id'] for r in records],
        device,
    )
    preds = probs.argmax(-1)
    labels = np.array([r['label'] for r in records])
    it = np.array([r['label'] in (LABEL2ID['GL_6220_IT_PERIPHERALS_OPEX'],
                                  LABEL2ID['GL_1400_IT_HARDWARE_CAPEX']) for r in records])
    near = np.array([
        amount_sensitive(r['metadata'].get('base_category'))
        and near_boundary(r['metadata'].get('base_category'), r['metadata']['net_amount_eur'], NEAR_REL)
        for r in records
    ])
    k, n = int((preds == labels).sum()), len(labels)
    return {
        'accuracy': k / n if n else None,
        'accuracy_ci95': wilson_ci(k, n),
        'it_accuracy': float((preds[it] == labels[it]).mean()) if it.any() else None,
        'it_count': int(it.sum()),
        'near_boundary_correct': int((preds[near] == labels[near]).sum()) if near.any() else 0,
        'near_boundary_count': int(near.sum()),
    }


def amount_sweep(model, tokenizer, records, scaler_mean, scaler_scale, device,
                 mono_tol: float = 0.05, step_tol_rel: float = DEFAULT_STEP_TOL_REL,
                 band_edges=None, return_curves: bool = False):
    """
    Re-score each amount-sensitive test description around each rule boundary with
    text and department fixed. The two accounts either side of the boundary should
    swap exactly once, at the boundary, with p(above) rising monotonically.

    Crossing error is measured for every case with a crossing (first upward
    crossing, interpolated in log-amount), not only for cases that happen to have
    exactly one flip. The conditioned version is kept separately as
    `single_flip_crossing_rel_error` for continuity with earlier runs.

    For a two-stage model (`band_edges` given) the price band token is recomputed
    for every swept amount; otherwise the fixed text would hide the amount.
    """
    cases = []
    single_flip_errors = []
    curve_data = []

    for r in records:
        cat = r['metadata'].get('base_category')
        for b in boundaries(cat):
            amounts = sweep_amounts(b.amount)
            feats = np.stack([amounts, np.log1p(amounts)], axis=1)
            feats = ((feats - scaler_mean) / scaler_scale).astype(np.float32)
            probs = predict_probs(
                model, tokenizer,
                sweep_texts(r['text'], amounts, band_edges),
                feats,
                [r['dept_id']] * len(amounts), device,
            )
            lo_id, hi_id = LABEL2ID[b.below], LABEL2ID[b.above]
            p_hi = probs[:, hi_id] / (probs[:, hi_id] + probs[:, lo_id] + 1e-9)
            want_hi = amounts >= b.amount
            strict = probs.argmax(-1) == np.where(want_hi, hi_id, lo_id)

            case = sweep_case_metrics(
                amounts, p_hi, b.amount,
                tight_rel=TIGHT_REL, step_tol_rel=step_tol_rel, mono_tol=mono_tol,
            )
            case['boundary'] = b.amount
            case['strict_correct'] = int(strict.sum())
            cases.append(case)
            curve_data.append({
                'amounts': amounts,
                'p_hi': p_hi,
                'max_prob': probs.max(-1),
                'boundary': b.amount,
            })
            if case['one_flip'] and case['crossing'] is not None:
                single_flip_errors.append(
                    abs(case['crossing'] - b.amount) / b.amount
                )

    agg = aggregate_sweep(cases, mono_tol=mono_tol, step_tol_rel=step_tol_rel,
                          tight_rel=TIGHT_REL)
    agg['single_flip_crossing_rel_error'] = (
        float(np.median(single_flip_errors)) if single_flip_errors else None
    )
    agg['single_flip_crossing_coverage'] = len(single_flip_errors) / max(len(cases), 1)
    if return_curves:
        return agg, curve_data
    return agg


def _panel_title_for_model(model_dir: Path, model, override: str = None) -> str:
    """Derive a descriptive panel title from the checkpoint path and model config."""
    if override:
        return override
    parts = set(model_dir.parts)
    if "D_ple_quantile_cf" in parts or model_dir.name == "multimodal-classifier":
        return "Default (Piecewise-Linear + Counterfactuals)"
    if "A_periodic_nocf" in parts:
        return "v0.3.0 Baseline (Periodic Encoding, No Counterfactuals)"
    enc = getattr(model, "num_encoding", "ple")
    enc_label = "Piecewise-Linear" if enc == "ple" else "Periodic Encoding"
    label = model_dir.parent.name if model_dir.name.startswith("seed") else model_dir.name
    return f"{label} ({enc_label})"


def plot_boundary_sweep(
    output_path: Path,
    primary_sweep: dict,
    primary_curves: list,
    primary_title: str = "Primary Model",
    compare_sweep: dict = None,
    compare_curves: list = None,
    compare_title: str = "Comparison Model",
    step_tol_rel: float = DEFAULT_STEP_TOL_REL,
):
    """Save a boundary-crossing figure showing P(above) across EUR amounts for each rule boundary."""
    if not primary_curves:
        return

    panels = []
    if compare_curves:
        panels.append((compare_title, compare_sweep or {}, compare_curves, "#c0392b", "#e74c3c"))
    panels.append((primary_title, primary_sweep or {}, primary_curves, "#0f62fe", "#4589ff"))

    distinct_boundaries = sorted({float(c['boundary']) for c in primary_curves})
    n_rows = len(distinct_boundaries)
    n_cols = len(panels)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(12.5 if n_cols == 2 else 6.8, 4.8 * n_rows),
        sharey=True,
        squeeze=False,
        dpi=180,
    )

    for row_idx, boundary in enumerate(distinct_boundaries):
        for col_idx, (title, sw, curves, dark_color, trace_color) in enumerate(panels):
            ax = axes[row_idx][col_idx]
            b_curves = [c for c in curves if abs(float(c['boundary']) - boundary) < 1e-6]
            if not b_curves:
                ax.set_visible(False)
                continue

            amounts = np.asarray(b_curves[0]['amounts'], dtype=float)
            mat = np.stack([np.asarray(c['p_hi'], dtype=float) for c in b_curves], axis=0)
            median_curve = np.median(mat, axis=0)
            q25 = np.percentile(mat, 25, axis=0)
            q75 = np.percentile(mat, 75, axis=0)

            ideal = (amounts >= boundary).astype(float)
            ax.step(
                amounts, ideal, where="post", color="#1f2937", linestyle=":", linewidth=1.4,
                label=f"Accounting rule (step at {boundary:g} EUR)", zorder=2,
            )

            tol_lo = boundary * (1 - step_tol_rel)
            tol_hi = boundary * (1 + step_tol_rel)
            ax.axvspan(
                tol_lo, tol_hi, color="#f59e0b", alpha=0.16,
                label=f"±{step_tol_rel:.1%} boundary band (manual review zone)",
            )
            ax.axvline(boundary, color="#111827", linestyle="--", linewidth=1.1, alpha=0.8)
            ax.axhline(0.5, color="#6b7280", linestyle="--", linewidth=0.9, alpha=0.6)

            for i, row in enumerate(mat):
                ax.plot(
                    amounts, row,
                    color=trace_color, alpha=0.20, linewidth=0.9,
                    label=f"{len(mat)} test items (text fixed)" if i == 0 else None,
                    zorder=3,
                )

            ax.fill_between(amounts, q25, q75, color=dark_color, alpha=0.18, zorder=4)
            ax.plot(
                amounts, median_curve, color=dark_color, linewidth=2.4,
                label="Median P(above) + IQR", zorder=5,
            )

            one_flip = (sw.get("exactly_one_flip_fraction") or 0.0) * 100
            non_mono = (sw.get("non_monotone_fraction") or 0.0) * 100
            cross_err = sw.get("median_crossing_rel_error")
            cross_str = f"{cross_err * 100:.1f}%" if cross_err is not None else "n/a"

            stats_box = (
                f"Single flip:     {one_flip:.0f}%\n"
                f"Non-monotone:    {non_mono:.0f}%\n"
                f"Crossing error:  {cross_str}"
            )
            ax.text(
                0.04, 0.78, stats_box,
                transform=ax.transAxes,
                fontsize=8.8,
                family="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#d1d5db", alpha=0.94),
                zorder=6,
            )

            panel_title = title if n_rows == 1 else f"{title} ({boundary:g} EUR boundary)"
            ax.set_title(panel_title, fontsize=10.5, fontweight="bold", pad=10)
            ax.set_xlabel("Invoice Net Amount (EUR, text & department held fixed)", fontsize=9.5)
            ax.set_xlim(amounts.min(), amounts.max())
            ax.set_ylim(-0.04, 1.04)
            ax.grid(True, linestyle="-", alpha=0.22)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[row_idx][0].set_ylabel(f"P(>= {boundary:g} EUR account)", fontsize=9.5)
        axes[row_idx][-1].legend(
            loc="lower right", frameon=True, fontsize=8.3, facecolor="white", framealpha=0.92
        )

    header_bound = (
        f"the {distinct_boundaries[0]:g} EUR Capitalization Boundary"
        if len(distinct_boundaries) == 1
        else "Rule Decision Boundaries"
    )
    fig.suptitle(
        f"Counterfactual Price Sweep Across {header_bound}",
        fontsize=12, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate multimodal ERP classifier")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output/evaluation-multimodal")
    parser.add_argument("--compare-model-dir", type=str, default=None,
                        help="Optional baseline checkpoint to plot alongside --model-dir in boundary_sweep.png")
    parser.add_argument("--primary-label", type=str, default=None,
                        help="Optional subplot title for --model-dir in the sweep plot")
    parser.add_argument("--compare-label", type=str, default=None,
                        help="Optional subplot title for --compare-model-dir in the sweep plot")
    parser.add_argument("--sweep-plot", type=str, default=None,
                        help="Filename for the sweep plot inside --output-dir (default: boundary_sweep.png when comparing, sweep_curve.png otherwise)")
    parser.add_argument("--mono-tolerance", type=float, default=0.05,
                        help="Max allowed negative step in p(above) before a case is non-monotone")
    parser.add_argument("--step-tolerance", type=float, default=DEFAULT_STEP_TOL_REL,
                        help="Relative band around the boundary excluded from tolerance-step accuracy")
    parser.add_argument("--skip-text-ablations", action="store_true",
                        help="Skip the price-stripped and constant-text diagnostic passes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Training seed, recorded in the summary (default: 42)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    console.print(f"Loading model from {args.model_dir}...")
    model, tokenizer, scaler_mean, scaler_scale = load_model(Path(args.model_dir), device)
    band_edges = load_price_band_edges(Path(args.model_dir))  # None for a single-stage model
    if band_edges is not None:
        console.print(f"  Two-stage model: price band edges {band_edges}")
    val_records = load_multimodal_dataset(Path(args.val_data))
    test_records = load_multimodal_dataset(Path(args.test_data))

    val_probs, val_preds, val_labels = run_split(model, tokenizer, val_records, device)
    probs, preds, labels = run_split(model, tokenizer, test_records, device)

    n = len(labels)
    k = int((preds == labels).sum())
    ci = wilson_ci(k, n)
    groups = record_groups(test_records)
    ci_clustered = cluster_bootstrap_ci((preds == labels).astype(float), groups)
    conf = probs.max(-1)
    correct = (preds == labels).astype(float)
    ece = expected_calibration_error(conf, correct)

    label_ids = sorted(set(labels) | set(preds))
    report = classification_report(
        labels, preds, labels=label_ids, target_names=[ID2LABEL[i] for i in label_ids],
        output_dict=True, zero_division=0,
    )

    console.print(f"\n[bold]Test accuracy:[/bold] {k/n:.4f}  (n={n}, {len(set(groups))} groups, "
                  f"Wilson CI [{ci[0]:.3f}, {ci[1]:.3f}], "
                  f"group-bootstrap CI [{ci_clustered[0]:.3f}, {ci_clustered[1]:.3f}])")
    console.print(f"[bold]Macro F1:[/bold] {report['macro avg']['f1-score']:.4f}   [bold]ECE:[/bold] {ece:.4f}")

    table = Table(title="Per-class metrics", show_header=True)
    for col in ("GL Account", "Precision", "Recall", "F1", "Support"):
        table.add_column(col, justify="left" if col == "GL Account" else "right")
    for i in label_ids:
        m = report[ID2LABEL[i]]
        table.add_row(ID2LABEL[i], f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1-score']:.3f}", str(int(m['support'])))
    console.print(table)

    # Slices
    slices = {}
    for key in ('language', 'corrupted', 'price_quoted'):
        for value in sorted({str(r['metadata'].get(key)) for r in test_records}):
            sel = [i for i, r in enumerate(test_records) if str(r['metadata'].get(key)) == value]
            ks = int((preds[sel] == labels[sel]).sum())
            slices[f"{key}={value}"] = {'count': len(sel), 'accuracy': ks / len(sel), 'ci95': wilson_ci(ks, len(sel))}
    console.print("\n[bold]Slices:[/bold]")
    for name, s in slices.items():
        console.print(f"  {name:<22} n={s['count']:<4} acc={s['accuracy']:.3f}  CI [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]")

    # Text ablations: price-stripped text and a constant description. These do not
    # change the checkpoint; they isolate how much of the score depends on the text.
    text_ablations = {}
    if not args.skip_text_ablations:
        variants = {
            'normal': [r['text'] for r in test_records],
            # Removes quoted prices only; the [PRICE_BAND] token of a two-stage model stays.
            'price_stripped': [scrub_prices(r['text']) for r in test_records],
            'constant_text': [
                sweep_texts('[VENDOR] Amazon Services Europe [DESC] office product',
                            [r['metadata']['net_amount_eur']], band_edges)[0]
                for r in test_records
            ],
        }
        for name, texts in variants.items():
            text_ablations[name] = evaluate_text_variant(model, tokenizer, test_records, texts, device)
        console.print("\n[bold]Text ablations (same checkpoint, changed text):[/bold]")
        for name, s in text_ablations.items():
            console.print(f"  {name:<18} acc={s['accuracy']:.3f}  IT={s['it_accuracy']:.3f}  "
                          f"near-boundary {s['near_boundary_correct']}/{s['near_boundary_count']}")

    # Boundary analysis on real test amounts
    thr = boundary_analysis(test_records, preds, labels)
    console.print("\n[bold]Rule-boundary analysis (real test items):[/bold]")
    for key in ('amount_sensitive_items', 'near_boundary_12.5pct', 'near_boundary_6.25pct'):
        t = thr[key]
        acc_s = f"{t['accuracy']:.3f}" if t['accuracy'] is not None else "n/a"
        console.print(f"  {key:<24} {t['correct']}/{t['count']}  acc={acc_s}")

    # Amount sweep
    sweep, primary_curves = amount_sweep(
        model, tokenizer, test_records, scaler_mean, scaler_scale, device,
        mono_tol=args.mono_tolerance, step_tol_rel=args.step_tolerance,
        band_edges=band_edges, return_curves=True,
    )
    console.print("\n[bold]Amount sweep (text/department fixed, amount varied around each boundary):[/bold]")
    console.print(f"  cases: {sweep['cases']}  point accuracy: {sweep['point_accuracy']:.3f}  "
                  f"strict (argmax): {sweep['strict_point_accuracy']:.3f}  "
                  f"tight (+/-6.25%): {sweep['tight_band_point_accuracy']:.3f}  "
                  f"tolerance (+/-{args.step_tolerance:.1%} band excluded): "
                  f"{sweep['tolerance_band_point_accuracy']:.3f}")
    console.print(f"  exactly one flip: {sweep['exactly_one_flip_fraction']:.1%}  "
                  f"non-monotone (tol {args.mono_tolerance}): {sweep['non_monotone_fraction']:.1%}  "
                  f"max negative step: {sweep['max_negative_step']:.3f}")
    if sweep['median_crossing_rel_error'] is not None:
        iqr = sweep['crossing_iqr_rel_error']
        console.print(f"  first-crossing error: {sweep['median_crossing_rel_error']:.2%} "
                      f"[IQR {iqr[0]:.2%}, {iqr[1]:.2%}]  "
                      f"coverage {sweep['crossing_coverage']:.0%}  "
                      f"censored {sweep['crossing_censored_fraction']:.0%}")
    else:
        console.print("  first-crossing error: n/a (no upward crossing observed)")
    if sweep['single_flip_crossing_rel_error'] is not None:
        console.print(f"  single-flip crossing error: {sweep['single_flip_crossing_rel_error']:.2%} "
                      f"(coverage {sweep['single_flip_crossing_coverage']:.0%})")

    primary_title = _panel_title_for_model(Path(args.model_dir), model, args.primary_label)
    compare_sweep, compare_curves, compare_title = None, None, "Comparison Model"
    if args.compare_model_dir:
        cmp_dir = Path(args.compare_model_dir)
        console.print(f"\nLoading comparison model from {cmp_dir} for sweep figure...")
        cmp_model, cmp_tok, cmp_mean, cmp_scale = load_model(cmp_dir, device)
        cmp_edges = load_price_band_edges(cmp_dir)
        compare_title = _panel_title_for_model(cmp_dir, cmp_model, args.compare_label)
        compare_sweep, compare_curves = amount_sweep(
            cmp_model, cmp_tok, test_records, cmp_mean, cmp_scale, device,
            mono_tol=args.mono_tolerance, step_tol_rel=args.step_tolerance,
            band_edges=cmp_edges, return_curves=True,
        )

    plot_name = args.sweep_plot or ("boundary_sweep.png" if args.compare_model_dir else "sweep_curve.png")
    sweep_plot_path = output_dir / plot_name
    plot_boundary_sweep(
        sweep_plot_path,
        primary_sweep=sweep,
        primary_curves=primary_curves,
        primary_title=primary_title,
        compare_sweep=compare_sweep,
        compare_curves=compare_curves,
        compare_title=compare_title,
        step_tol_rel=args.step_tolerance,
    )
    console.print(f"Saved sweep plot to {sweep_plot_path}")

    # Confidence cut-off: chosen on val, reported once on test
    val_conf = val_probs.max(-1)
    val_correct = (val_preds == val_labels).astype(float)
    val_curve, chosen = risk_coverage(val_conf, val_correct)
    test_curve, _ = risk_coverage(conf, correct)
    if chosen:
        mask = conf >= chosen['cutoff']
        abstain = {
            'target_accuracy': 0.99, 'target_met_on_val': True, 'cutoff_selected_on_val': chosen['cutoff'],
            'val_coverage': chosen['coverage'], 'test_coverage': float(mask.mean()),
            'test_accuracy_on_accepted': float(correct[mask].mean()) if mask.any() else None,
        }
        console.print(f"\n[bold]Confidence cut-off (selected on val for 99% accuracy):[/bold] {chosen['cutoff']:.2f} -> "
                      f"test coverage {abstain['test_coverage']:.1%}, accuracy on accepted {abstain['test_accuracy_on_accepted']:.3f}")
    else:
        abstain = {'target_accuracy': 0.99, 'target_met_on_val': False}
        console.print("\n[bold]Confidence cut-off:[/bold] 99% accuracy not reachable on val at any cut-off")
    abstain['val_risk_coverage'], abstain['test_risk_coverage'] = val_curve, test_curve

    cm = confusion_matrix(labels, preds, labels=label_ids)
    results = {
        'seed': args.seed,
        'price_band_edges': band_edges,
        'n_test': n,
        'n_groups': len(set(groups)),
        'accuracy': k / n,
        'accuracy_ci95': ci,
        'accuracy_ci95_group_bootstrap': ci_clustered,
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'ece': ece,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'slices': slices,
        'text_ablations': text_ablations,
        'boundary_analysis': thr,
        'amount_sweep': sweep,
        'abstention': abstain,
    }
    with open(output_dir / "evaluation_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    console.print(f"\nSaved {output_dir / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
