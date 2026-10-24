"""
Aggregate per-seed evaluation summaries into markdown tables (and optionally a CSV).

Metrics that condition on a post-treatment event ("exactly one flip") are
reported with their coverage, and the unconditional first-crossing error is
the headline.

Usage:
    python scripts/summarize_ablation.py \
        --root output/ablation \
        --output output/evaluation-multimodal/ablation_summary.md
"""

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np


def _get(d, *keys, default=None):
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def load_rows(root):
    rows = {}
    for f in sorted(glob.glob(f"{root}/*/eval_seed*/evaluation_summary.json")):
        config = Path(f).parts[-3]
        d = json.load(open(f))
        match = re.search(r"eval_seed(\d+)", str(f))
        d["_seed"] = int(match.group(1)) if match else None
        rows.setdefault(config, []).append(d)
    return rows


def _fmt_pct(x, digits=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.{digits}f}%"


def _fmt_num(x, digits=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def seed_row(config, seed, d):
    sw = d.get("amount_sweep", {})
    ba = d.get("boundary_analysis", {})
    near = ba.get("near_boundary_12.5pct", {})
    ta = d.get("text_ablations", {})
    ci = d.get("accuracy_ci95") or [None, None]
    gci = d.get("accuracy_ci95_group_bootstrap") or [None, None]
    iqr = sw.get("crossing_iqr_rel_error") or [None, None]
    return {
        "config": config,
        "seed": seed,
        "n_test": d.get("n_test"),
        "n_groups": d.get("n_groups"),
        "accuracy": d.get("accuracy"),
        "acc_ci_lo": ci[0],
        "acc_ci_hi": ci[1],
        "acc_group_ci_lo": gci[0],
        "acc_group_ci_hi": gci[1],
        "macro_f1": d.get("macro_f1"),
        "ece": d.get("ece"),
        "near_correct": near.get("correct"),
        "near_count": near.get("count"),
        "sweep_cases": sw.get("cases"),
        "sweep_point_accuracy": sw.get("point_accuracy"),
        "sweep_strict_accuracy": sw.get("strict_point_accuracy"),
        "sweep_tight_accuracy": sw.get("tight_band_point_accuracy"),
        "sweep_tolerance_accuracy": sw.get("tolerance_band_point_accuracy"),
        "sweep_tolerance_step_fraction": sw.get("tolerance_step_fraction"),
        "sweep_one_flip_fraction": sw.get("exactly_one_flip_fraction"),
        "sweep_non_monotone_fraction": sw.get("non_monotone_fraction"),
        "sweep_max_negative_step": sw.get("max_negative_step"),
        "crossing_median": sw.get("median_crossing_rel_error"),
        "crossing_q25": iqr[0],
        "crossing_q75": iqr[1],
        "crossing_coverage": sw.get("crossing_coverage"),
        "crossing_censored": sw.get("crossing_censored_fraction"),
        "single_flip_crossing": sw.get("single_flip_crossing_rel_error"),
        "single_flip_coverage": sw.get("single_flip_crossing_coverage"),
        "it_accuracy": _get(ta, "normal", "it_accuracy"),
        "price_stripped_accuracy": _get(ta, "price_stripped", "accuracy"),
        "constant_text_accuracy": _get(ta, "constant_text", "accuracy"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/ablation")
    ap.add_argument("--output", default="output/evaluation-multimodal/ablation_summary.md")
    ap.add_argument("--csv", default=None, help="Optional path to write per-seed CSV rows")
    args = ap.parse_args()

    rows = load_rows(args.root)
    if not rows:
        raise SystemExit(f"No evaluation summaries under {args.root}/*/eval_seed*/")

    if args.csv:
        all_rows = []
        for config, summaries in rows.items():
            for i, d in enumerate(sorted(summaries, key=lambda x: x.get("seed") or x.get("_seed") or 0)):
                seed = d.get("seed")
                if seed is None:
                    seed = d.get("_seed", i)
                all_rows.append(seed_row(config, seed, d))

        out_csv = Path(args.csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(all_rows)

    lines = [
        "| Config | Seeds | Test acc (seed sd) | Macro F1 | ECE | Real near-boundary |",
        "|---|---|---|---|---|---|",
    ]
    for config, summaries in sorted(rows.items()):
        acc = np.array([d["accuracy"] for d in summaries], dtype=float)
        f1 = np.array([d["macro_f1"] for d in summaries], dtype=float)
        ece = np.array([d["ece"] for d in summaries], dtype=float)
        near = [
            (d["boundary_analysis"]["near_boundary_12.5pct"]["correct"],
             d["boundary_analysis"]["near_boundary_12.5pct"]["count"])
            for d in summaries
        ]
        near_str = ", ".join(f"{k}/{n}" for k, n in near)
        lines.append(
            f"| {config} | {len(summaries)} | {_fmt_pct(acc.mean())} ± {_fmt_pct(acc.std())} "
            f"| {f1.mean():.3f} | {ece.mean():.3f} | {near_str} |"
        )

    lines += [
        "",
        "| Config | Sweep point | Strict | Tight ±6.25% | Tolerance step | One flip | Non-monotone | Max neg. step | First-crossing error (seed range) | Coverage |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for config, summaries in sorted(rows.items()):
        def mean(key):
            vals = [d["amount_sweep"].get(key) for d in summaries]
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else None

        crossing = mean("median_crossing_rel_error")
        seed_crossings = [
            d["amount_sweep"].get("median_crossing_rel_error") for d in summaries
        ]
        seed_crossings = [x for x in seed_crossings if x is not None]
        coverage = mean("crossing_coverage")
        if crossing is not None and seed_crossings:
            crossing_str = (
                f"{crossing*100:.1f}% "
                f"[{min(seed_crossings)*100:.1f}, {max(seed_crossings)*100:.1f}]"
            )
        else:
            crossing_str = "n/a"
        lines.append(
            f"| {config} | {_fmt_num(mean('point_accuracy'))} | {_fmt_num(mean('strict_point_accuracy'))} "
            f"| {_fmt_num(mean('tight_band_point_accuracy'))} "
            f"| {_fmt_pct(mean('tolerance_step_fraction'), 0)} "
            f"| {_fmt_pct(mean('exactly_one_flip_fraction'), 0)} "
            f"| {_fmt_pct(mean('non_monotone_fraction'), 0)} "
            f"| {_fmt_num(mean('max_negative_step'))} "
            f"| {crossing_str} | {_fmt_pct(coverage, 0)} |"
        )

    lines += [
        "",
        "Notes:",
        "- `±` is the training-seed standard deviation on one fixed test set, not test-set uncertainty.",
        "- First-crossing error is the first upward 0.5 crossing, interpolated in log-amount, over every case that has a crossing; `Coverage` is the fraction of cases with a crossing. It is not conditioned on the curve having exactly one flip. The bracketed range is the min/max of the per-seed medians, not a confidence interval; for a noisy curve (config A) the first crossing is an early noise crossing.",
        "- `Tolerance step` requires the correct account at every amount outside a ±2% band around the boundary. The old `perfect step` also required the ±0.001% points and is 0% by construction.",
        "- `Non-monotone` means any negative step in p(above) larger than 0.05; `Max neg. step` reports the largest such drop.",
        "- `Real near-boundary` counts actual test items within ±12.5% of a rule boundary (not synthetic sweep points); it is shown per seed.",
    ]
    if args.csv:
        lines.append(f"- Raw per-seed rows: `{args.csv}`.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
