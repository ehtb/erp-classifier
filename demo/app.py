#!/usr/bin/env python3
"""
Interactive browser demo for the ERP GL Account Classifier.

Run locally:
    pip install -e ".[demo]"
    python demo/app.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classify_erp import DEFAULT_CONFIDENCE_CUTOFF, DEFAULT_MODEL_DIR, ERPClassifier  # noqa: E402
from src.boundaries import price_band_edges  # noqa: E402
from src.config import LABEL2ID  # noqa: E402
from src.metrics import DEFAULT_STEP_TOL_REL  # noqa: E402
from src.model import predict_probs  # noqa: E402
from src.text import scrub_prices, sweep_texts  # noqa: E402

PRESET_EXAMPLES = [
    ["Amazon Services Europe", "Dell UltraSharp U2723DE 27 Monitor", "IT_OPS", 750.0],
    ["Amazon Services Europe", "Dell UltraSharp U2723DE 27 Monitor", "IT_OPS", 850.0],
    ["Amazon Services Europe", "Dell UltraSharp U2723DE 27 Monitor", "IT_OPS", 800.0],
    ["AMZN Mktp DE", "DELL LAT 5540 CTO I7/16/512GB/15.6FHD", "ENG", 1190.0],
    ["Amazon.nl", "Lavazza Qualita Rossa Koffiebonen 1kg", "FACILITIES", 18.50],
    ["Amazon EU S.a.r.l.", "Credit note adjustment", "UNK", 0.0],
]


def compute_item_sweep_figure(
    classifier: ERPClassifier,
    vendor: str,
    description: str,
    department: str,
    current_amount: float,
    step_tol_rel: float = DEFAULT_STEP_TOL_REL,
):
    """Plot P(CAPEX) and P(OPEX) across EUR amounts for the current item."""
    edges = classifier.band_edges or list(price_band_edges())
    primary_edge = float(edges[0]) if edges else 800.0

    lo = min(primary_edge * 0.6, max(0.0, float(current_amount) * 0.85))
    hi = max(primary_edge * 1.55, float(current_amount) * 1.10)
    amounts = np.linspace(lo, hi, 75)

    base_text = f"[VENDOR] {vendor} [DESC] {scrub_prices(description)}"
    texts = sweep_texts(base_text, amounts, classifier.band_edges)
    feats = np.stack([amounts, np.log1p(np.maximum(0.0, amounts))], axis=1)
    feats = ((feats - classifier.scaler_mean) / classifier.scaler_scale).astype(np.float32)
    dept_id = classifier.dept_vocab.get(department.upper(), 0)

    probs = predict_probs(
        classifier.model,
        classifier.tokenizer,
        texts,
        feats,
        [dept_id] * len(amounts),
        classifier.device,
    )
    opex_id = LABEL2ID["GL_6220_IT_PERIPHERALS_OPEX"]
    capex_id = LABEL2ID["GL_1400_IT_HARDWARE_CAPEX"]

    p_opex = probs[:, opex_id]
    p_capex = probs[:, capex_id]
    max_conf = probs.max(-1)

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=140)
    for idx, edge in enumerate(edges):
        tol_lo = edge * (1.0 - step_tol_rel)
        tol_hi = edge * (1.0 + step_tol_rel)
        ax.axvspan(
            tol_lo,
            tol_hi,
            color="#f59e0b",
            alpha=0.16,
            label=f"±{step_tol_rel:.1%} boundary zone" if idx == 0 else None,
        )
        ax.axvline(
            edge,
            color="#111827",
            linestyle="--",
            linewidth=1.2,
            label=f"{edge:g} EUR accounting rule" if idx == 0 else None,
        )

    ax.axhline(
        classifier.confidence_cutoff,
        color="#6b7280",
        linestyle=":",
        linewidth=1.0,
        label=f"{classifier.confidence_cutoff:.0%} auto-post cutoff",
    )

    ax.plot(amounts, p_opex, color="#0891b2", linewidth=2.0, label="P(GL_6220 OPEX)")
    ax.plot(amounts, p_capex, color="#2563eb", linewidth=2.2, label="P(GL_1400 CAPEX)")
    ax.plot(amounts, max_conf, color="#9ca3af", linewidth=1.1, linestyle="-.", label="Top-1 confidence")

    if lo <= current_amount <= hi:
        ax.axvline(
            current_amount,
            color="#dc2626",
            linewidth=1.6,
            label=f"Selected: {current_amount:.0f} EUR",
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Net Amount (EUR, text & department held fixed)")
    ax.set_ylabel("Probability")
    ax.set_title(
        f"Counterfactual Price Sweep ({lo:.0f} to {hi:.0f} EUR)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center right", fontsize=8.0, frameon=True)
    fig.tight_layout()
    return fig


def evaluate_transaction(
    classifier: ERPClassifier,
    vendor: str,
    description: str,
    department: str,
    amount_eur: float,
) -> tuple[str, dict[str, float], str, plt.Figure]:
    """Run prediction, format routing summary and diagnostics, and build the sweep plot."""
    res = classifier.predict(vendor, description, department, float(amount_eur))
    gl = res["gl_account"]
    conf = res["confidence"]
    cutoff = res["cutoff"]
    routing = res["routing"]
    diag = res["diagnostics"]

    if routing == "AUTO_POST":
        badge = f"AUTO-POST (confidence {conf:.1%} >= {cutoff:.0%} cutoff)"
    else:
        badge = f"ROUTE TO MANUAL REVIEW (confidence {conf:.1%} < {cutoff:.0%} cutoff)"

    summary_md = (
        f"### Predicted account: `{gl}`\n"
        f"- **Confidence:** {conf:.1%}\n"
        f"- **Routing action:** {badge}\n"
    )

    encoder_text = diag.get("encoder_text", f"[VENDOR] {vendor} [DESC] {diag['scrubbed_description']}")
    diag_md = (
        f"**Text sent to encoder:** `{encoder_text}`  \n"
        f"**FiLM gate modulation:** scale mean `{diag['film_scale_mean']:.2f}`, "
        f"shift norm `{diag['film_shift_norm']:.2f}`, "
        f"relative `[CLS]` shift `{diag['relative_text_modulation']:.1%}`"
    )

    fig = compute_item_sweep_figure(classifier, vendor, description, department, float(amount_eur))
    return summary_md, res["all_predictions"], diag_md, fig


def build_demo(classifier: ERPClassifier):
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit(
            "Gradio is required for the browser demo. Install it with: pip install -e '.[demo]'"
        ) from exc

    departments = [d for d in classifier.dept_vocab.keys()]
    edges = classifier.band_edges or list(price_band_edges())
    edge_label = f"{edges[0]:g} EUR" if edges else "the rule boundary"

    with gr.Blocks(title="ERP GL Account Classifier") as demo:
        gr.Markdown(
            "# ERP GL Account Classifier\n"
            "Predicts general ledger accounts from invoice text, department, and net EUR amount. "
            f"Slide the amount across **{edge_label}** on an IT hardware item to see the model switch "
            "from operating expense (`GL_6220`) to capital expenditure (`GL_1400`) and flag "
            "borderline amounts for manual review."
        )

        with gr.Row():
            with gr.Column(scale=1):
                vendor_in = gr.Textbox(label="Vendor", value="Amazon Services Europe")
                desc_in = gr.Textbox(
                    label="Item Description",
                    value="Dell UltraSharp U2723DE 27 Monitor",
                )
                dept_in = gr.Dropdown(
                    choices=departments,
                    value="IT_OPS",
                    label="Department",
                )
                amount_in = gr.Slider(
                    minimum=0.0,
                    maximum=2000.0,
                    step=5.0,
                    value=750.0,
                    label="Net Amount (EUR)",
                )
                run_btn = gr.Button("Classify Invoice Line", variant="primary")

            with gr.Column(scale=1):
                summary_out = gr.Markdown()
                probs_out = gr.Label(num_top_classes=5, label="GL Account Probabilities")
                diag_out = gr.Markdown()

        sweep_plot = gr.Plot(label="Counterfactual Price Sweep")

        gr.Examples(
            examples=PRESET_EXAMPLES,
            inputs=[vendor_in, desc_in, dept_in, amount_in],
            label="Preset invoice lines (click to load)",
        )

        inputs = [vendor_in, desc_in, dept_in, amount_in]
        outputs = [summary_out, probs_out, diag_out, sweep_plot]

        run_btn.click(
            fn=lambda v, d, dp, a: evaluate_transaction(classifier, v, d, dp, a),
            inputs=inputs,
            outputs=outputs,
        )
        amount_in.release(
            fn=lambda v, d, dp, a: evaluate_transaction(classifier, v, d, dp, a),
            inputs=inputs,
            outputs=outputs,
        )
        demo.load(
            fn=lambda v, d, dp, a: evaluate_transaction(classifier, v, d, dp, a),
            inputs=inputs,
            outputs=outputs,
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch the ERP Classifier browser demo")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cutoff", type=float, default=DEFAULT_CONFIDENCE_CUTOFF)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    classifier = ERPClassifier(args.model_dir, confidence_cutoff=args.cutoff)
    demo = build_demo(classifier)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
