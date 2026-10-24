"""
Reproduce the A-E encoding/counterfactual ablation and the class-weight ablation
on one deterministic split. `--preset band` compares D with F, the two-stage model
that gets a rule-derived price band token (a with-prior upper bound). It writes its own
band_ablation.md; the all-configs table is `summarize_ablation.py --root output/ablation`.

Each configuration gets its own dataset directory (scaler and bin edges are fit
on its own training data) unless it reuses an identical dataset configuration,
and the split itself is group-based and independent of the model seed. Seeds
vary only the training run.

Usage:
    python scripts/run_ablation.py --seeds 42 43 44
    python scripts/run_ablation.py --preset weights
    python scripts/run_ablation.py --configs D_ple_quantile_cf --seeds 42
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

CONFIGS = {
    "A_periodic_nocf":     {"counterfactuals": 0, "encoding": "periodic", "bins": "quantile", "weight_mode": "cf",       "seeds": [42, 43, 44]},
    "B_ple_quantile_nocf": {"counterfactuals": 0, "encoding": "ple",      "bins": "quantile", "weight_mode": "cf",       "seeds": [42, 43, 44]},
    "C_periodic_cf":       {"counterfactuals": 4, "encoding": "periodic", "bins": "quantile", "weight_mode": "cf",       "seeds": [42, 43, 44]},
    "D_ple_quantile_cf":   {"counterfactuals": 4, "encoding": "ple",      "bins": "quantile", "weight_mode": "cf",       "seeds": [42, 43, 44]},
    "E_ple_tree_cf":       {"counterfactuals": 4, "encoding": "ple",      "bins": "tree",     "weight_mode": "cf",       "seeds": [42, 43, 44]},
    "F_ple_quantile_cf_band": {"counterfactuals": 4, "encoding": "ple", "bins": "quantile", "weight_mode": "cf", "seeds": [42, 43, 44], "price_band": True},
    "C_balanced":          {"counterfactuals": 4, "encoding": "periodic", "bins": "quantile", "weight_mode": "balanced", "seeds": [42],         "data_dir": "output/multimodal"},
    "D_balanced":          {"counterfactuals": 4, "encoding": "ple",      "bins": "quantile", "weight_mode": "balanced", "seeds": [42],         "data_dir": "output/multimodal"},
    "D_none":              {"counterfactuals": 4, "encoding": "ple",      "bins": "quantile", "weight_mode": "none",     "seeds": [42, 43, 44], "data_dir": "output/multimodal"},
}

PRESETS = {
    "main": ["A_periodic_nocf", "B_ple_quantile_nocf", "C_periodic_cf", "D_ple_quantile_cf", "E_ple_tree_cf"],
    "band": ["D_ple_quantile_cf", "F_ple_quantile_cf_band"],
    "weights": ["C_balanced", "D_balanced", "D_none"],
    "all": list(CONFIGS),
}


def run(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(str(c) for c in cmd))
    with open(log, "w") as f:
        subprocess.run([str(c) for c in cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="main",
                    help="Configuration group to run when --configs is not specified")
    ap.add_argument("--configs", nargs="+", default=None, choices=list(CONFIGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Override per-config default seeds")
    ap.add_argument("--weight-mode", choices=["cf", "balanced", "none"], default=None,
                    help="Override per-config class-weight mode")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--model-root", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    is_weights = args.preset == "weights" and args.configs is None
    configs = args.configs or PRESETS[args.preset]
    data_root_rel = args.data_root or ("output/weight_ablation" if is_weights else "output/ablation")
    model_root_rel = args.model_root or ("models/weight_ablation" if is_weights else "models/ablation")
    is_band = args.preset == "band" and args.configs is None
    summary_rel = args.summary or (
        "output/evaluation-multimodal/weight_ablation.md" if is_weights
        else "output/evaluation-multimodal/band_ablation.md" if is_band
        else "output/evaluation-multimodal/ablation_summary.md"
    )

    data_root = ROOT / data_root_rel
    model_root = ROOT / model_root_rel
    data_root.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs/ablation").mkdir(parents=True, exist_ok=True)

    for name in configs:
        cfg = CONFIGS[name]
        seeds = args.seeds if args.seeds is not None else cfg["seeds"]
        weight_mode = args.weight_mode or cfg["weight_mode"]
        data_dir = ROOT / cfg["data_dir"] if "data_dir" in cfg else data_root / name

        if not args.skip_data and (args.force or not (data_dir / "train_multimodal.jsonl").exists()):
            run([
                PY, "scripts/build_dataset.py",
                "--input", "output/03_filtered.jsonl",
                "--output-dir", data_dir,
                "--counterfactuals-per-record", cfg["counterfactuals"],
                "--bin-strategy", cfg["bins"],
                "--seed", 42,
            ] + (["--price-band"] if cfg.get("price_band") else []), ROOT / f"logs/ablation/{name}_build.log")

        for seed in seeds:
            model_dir = model_root / name / f"seed{seed}"
            if not args.skip_train and (args.force or not (model_dir / "config.json").exists()):
                run([
                    PY, "train.py",
                    "--train-data", data_dir / "train_multimodal.jsonl",
                    "--val-data", data_dir / "val_multimodal.jsonl",
                    "--output-dir", model_dir,
                    "--num-epochs", args.epochs,
                    "--seed", seed,
                    "--num-encoding", cfg["encoding"],
                    "--class-weight-mode", weight_mode,
                ], ROOT / f"logs/ablation/{name}_seed{seed}_train.log")

            eval_dir = data_root / name / f"eval_seed{seed}"
            if not args.skip_eval and (args.force or not (eval_dir / "evaluation_summary.json").exists()):
                run([
                    PY, "scripts/evaluate.py",
                    "--model-dir", model_dir,
                    "--val-data", data_dir / "val_multimodal.jsonl",
                    "--test-data", data_dir / "test_multimodal.jsonl",
                    "--output-dir", eval_dir,
                    "--seed", seed,
                ], ROOT / f"logs/ablation/{name}_seed{seed}_eval.log")

    summarize_cmd = [
        PY, "scripts/summarize_ablation.py",
        "--root", data_root_rel,
        "--output", summary_rel,
    ]
    if is_band:
        # Own file, limited to D and F, so this never overwrites the A-E table.
        summarize_cmd += ["--configs", *PRESETS["band"]]
    if args.csv:
        summarize_cmd += ["--csv", args.csv]
    run(summarize_cmd, ROOT / "logs/ablation/summarize.log")


if __name__ == "__main__":
    main()
