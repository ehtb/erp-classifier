#!/usr/bin/env python3
"""
ERP GL Account Classifier - Command Line Tool

Usage:
    # Interactive mode
    python classify_erp.py

    # One-shot mode
    python classify_erp.py \
        --vendor "Amazon Services Europe" \
        --description "Dell UltraSharp U2723DE 27 Monitor" \
        --department IT_OPS \
        --amount 450

    # Price sweep across the 800 EUR capitalization boundary
    python classify_erp.py --sweep 650:950:25

    # Quick mode (minimal prompts)
    python classify_erp.py --quick
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import FloatPrompt, Prompt
from rich.table import Table
from transformers import AutoTokenizer

from src.boundaries import price_band_edges
from src.config import ID2LABEL, LABEL2ID
from src.model import ERPClassifierModel, get_device
from src.text import load_price_band_edges, scrub_prices, set_price_band

console = Console()

DEFAULT_MODEL_DIR = "models/multimodal-classifier"
FALLBACK_MODEL_DIRS = (
    "models/multimodal-v2",
    "models/ablation/D_ple_quantile_cf/seed42",
)
DEFAULT_CONFIDENCE_CUTOFF = 0.90


class ModelLoadError(RuntimeError):
    """Raised when a checkpoint or one of its preprocessing artifacts is missing/invalid."""


def parse_sweep_spec(spec: str) -> list[float]:
    """Parse a sweep specification ('start:stop:step' or 'a,b,c') into EUR amounts."""
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty sweep specification")
    if ":" in spec:
        parts = [float(x) for x in spec.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 25.0
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"Expected start:stop or start:stop:step, got {spec!r}")
        if step <= 0 or stop < start:
            raise ValueError(f"Invalid sweep range: {spec!r}")
        n_steps = int(math.floor((stop - start) / step + 1e-9))
        amounts = [round(start + i * step, 2) for i in range(n_steps + 1)]
        if amounts[-1] < round(stop, 2):
            amounts.append(round(stop, 2))
        return amounts

    amounts = [round(float(x.strip()), 2) for x in spec.split(",") if x.strip()]
    if not amounts:
        raise ValueError(f"Sweep specification produced no amounts: {spec!r}")
    return amounts


class ERPClassifier:
    """Classifier with loaded model and preprocessing artifacts."""

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        confidence_cutoff: float = DEFAULT_CONFIDENCE_CUTOFF,
        device: str | torch.device | None = None,
    ):
        if not (0.0 <= float(confidence_cutoff) <= 1.0):
            raise ValueError(f"confidence_cutoff must be between 0.0 and 1.0, got {confidence_cutoff!r}")
        self.model_dir = self._resolve_model_dir(Path(model_dir))
        self.confidence_cutoff = float(confidence_cutoff)
        self.device = torch.device(device) if device is not None else get_device()
        self.model = None
        self.tokenizer = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.dept_vocab = None
        self.id2dept = None

        self._load_model()

    @staticmethod
    def _resolve_model_dir(requested: Path) -> Path:
        if requested.exists():
            return requested
        if requested == Path(DEFAULT_MODEL_DIR):
            for candidate in FALLBACK_MODEL_DIRS:
                cand_path = Path(candidate)
                if cand_path.exists():
                    console.print(
                        f"[yellow]Warning: {requested} not found; "
                        f"falling back to {cand_path}.[/yellow]"
                    )
                    return cand_path
        return requested

    def _artifact(self, name: str) -> Path:
        """Preprocessing artifact saved next to the checkpoint.

        No fallback: mixing the scaler/vocabulary of a different dataset build with
        this checkpoint silently produces wrong probabilities.
        """
        path = self.model_dir / name
        if not path.exists():
            raise ModelLoadError(
                f"required artifact not found: {path}. The checkpoint must ship "
                "with its own scaler/vocabulary. Retrain with train.py, which "
                "copies them next to the weights."
            )
        return path

    def _load_model(self):
        """Load model, tokenizer, and auxiliary files."""
        console.print(f"\n[bold cyan]Loading ERP Classifier ({self.model_dir})...[/bold cyan]")

        if not self.model_dir.exists():
            available = []
            models_dir = Path("models")
            if models_dir.exists():
                available = [
                    str(p) for p in sorted(models_dir.iterdir()) if p.is_dir()
                ]
            available_text = (
                "\nAvailable models:\n  " + "\n  ".join(available)
                if available
                else ""
            )
            raise ModelLoadError(
                f"Model directory not found: {self.model_dir}{available_text}\n"
                f"Train a model with: python train.py "
                f"--train-data output/multimodal/train_multimodal.jsonl "
                f"--val-data output/multimodal/val_multimodal.jsonl "
                f"--output-dir {self.model_dir}"
            )

        with console.status("[bold green]Loading tokenizer..."):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        scaler_path = self._artifact("scaler.json")
        with open(scaler_path) as f:
            scaler_params = json.load(f)
            self.scaler_mean = np.array(scaler_params['mean'])
            self.scaler_scale = np.array(scaler_params['scale'])

        # Two-stage checkpoints ship price band edges; single-stage ones do not.
        self.band_edges = load_price_band_edges(self.model_dir)

        vocab_path = self._artifact("dept_vocab.json")
        with open(vocab_path) as f:
            vocab_data = json.load(f)
            self.dept_vocab = vocab_data['dept2id']
            self.id2dept = {int(k): v for k, v in vocab_data['id2dept'].items()}

        with console.status("[bold green]Loading model weights..."):
            self.model = ERPClassifierModel.from_pretrained(self.model_dir)
            self.model.to(self.device).eval()

        if self.model.num_labels != len(ID2LABEL):
            raise ModelLoadError(
                f"Checkpoint has {self.model.num_labels} labels, "
                f"but the code expects {len(ID2LABEL)}."
            )
        if (
            len(self.scaler_mean) != self.model.num_continuous
            or len(self.scaler_scale) != self.model.num_continuous
        ):
            raise ModelLoadError(
                f"Scaler has {len(self.scaler_mean)} features, but the model "
                f"expects {self.model.num_continuous}."
            )
        if max(self.dept_vocab.values(), default=0) >= self.model.num_depts:
            raise ModelLoadError(
                f"Department vocabulary has id {max(self.dept_vocab.values())}, "
                f"but the model only has {self.model.num_depts} embeddings."
            )

        console.print(f"[bold green]Model loaded on {self.device}.[/bold green]\n")

    def predict(
        self,
        vendor: str,
        description: str,
        department: str,
        net_amount_eur: float,
    ) -> dict:
        """
        Predict GL account and confidence routing for a transaction.

        Returns:
            dict with keys: gl_account, confidence, routing, cutoff, all_predictions, diagnostics
        """
        scrubbed_desc = scrub_prices(description)
        text = f"[VENDOR] {vendor} [DESC] {scrubbed_desc}"
        if self.band_edges is not None:
            text = set_price_band(text, net_amount_eur, self.band_edges)
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=getattr(self.model, "max_seq_length", 128),
            padding='max_length',
            return_tensors='pt',
        )

        # Clamp at 0 so negative credit-note amounts do not NaN in log1p
        safe_amount = max(0.0, float(net_amount_eur))
        num_features = np.array([net_amount_eur, np.log1p(safe_amount)])
        num_features = (num_features - self.scaler_mean) / self.scaler_scale

        dept_id = self.dept_vocab.get(department.upper(), 0)  # 0 = UNK

        diag = self.model.forward_with_diagnostics(
            input_ids=encoding['input_ids'].to(self.device),
            attention_mask=encoding['attention_mask'].to(self.device),
            num_features=torch.from_numpy(num_features).unsqueeze(0).float().to(self.device),
            dept_id=torch.tensor([dept_id], dtype=torch.long, device=self.device),
        )

        probs = diag['probs'].cpu()
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[0, pred_idx].item())
        routing = "AUTO_POST" if confidence >= self.confidence_cutoff else "MANUAL_REVIEW"

        all_predictions = {
            ID2LABEL[i]: float(probs[0, i])
            for i in range(len(ID2LABEL))
        }

        return {
            'gl_account': ID2LABEL[pred_idx],
            'confidence': confidence,
            'routing': routing,
            'cutoff': self.confidence_cutoff,
            'all_predictions': all_predictions,
            'diagnostics': {
                'scrubbed_description': scrubbed_desc,
                'encoder_text': text,
                'film_scale_mean': float(diag['film_scale_mean'][0].cpu().item()),
                'film_shift_norm': float(diag['film_shift_norm'][0].cpu().item()),
                'relative_text_modulation': float(diag['relative_text_modulation'][0].cpu().item()),
            },
        }


def display_prediction(result: dict, show_all: bool = False):
    """Display prediction results and routing status."""
    gl_account = result['gl_account']
    confidence = result['confidence']
    cutoff = result.get('cutoff', DEFAULT_CONFIDENCE_CUTOFF)
    routing = result.get('routing', 'AUTO_POST' if confidence >= cutoff else 'MANUAL_REVIEW')

    if routing == "AUTO_POST":
        color = "green"
        routing_line = f"[bold green]AUTO-POST[/bold green] (confidence >= {cutoff:.0%} configured cutoff)"
    elif confidence >= 0.70:
        color = "yellow"
        routing_line = f"[bold yellow]MANUAL REVIEW[/bold yellow] (below {cutoff:.0%} auto-post cutoff)"
    else:
        color = "red"
        routing_line = f"[bold red]MANUAL REVIEW[/bold red] (low confidence, below {cutoff:.0%} cutoff)"

    console.print(Panel(
        f"[bold {color}]{gl_account}[/bold {color}]\n"
        f"Confidence: {confidence:.1%}\n"
        f"Routing:    {routing_line}",
        title="[bold]Prediction[/bold]",
        border_style=color,
    ))

    if show_all or confidence < 0.7:
        console.print("\n[bold]Top GL Account Probabilities:[/bold]")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("GL Account", style="cyan")
        table.add_column("Probability", justify="right")
        table.add_column("Bar", style="blue")

        sorted_preds = sorted(
            result['all_predictions'].items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for gl, prob in sorted_preds[:5]:
            bar = "█" * int(prob * 20)
            table.add_row(
                gl.replace("GL_", ""),
                f"{prob:.1%}",
                bar,
            )

        console.print(table)

        diag = result.get('diagnostics')
        if diag and show_all:
            console.print(
                f"[dim]FiLM modulation: scale mean={diag['film_scale_mean']:.2f}, "
                f"shift norm={diag['film_shift_norm']:.2f}, "
                f"text shift={diag['relative_text_modulation']:.1%}[/dim]"
            )


def sweep_mode(
    classifier: ERPClassifier,
    vendor: str,
    description: str,
    department: str,
    amounts: list[float],
):
    """Run a counterfactual amount sweep holding vendor, description, and department fixed."""
    console.print(Panel(
        f"[bold]Vendor:[/bold]      {vendor}\n"
        f"[bold]Description:[/bold] {description}\n"
        f"[bold]Department:[/bold]  {department}\n"
        f"[bold]Sweep:[/bold]       {len(amounts)} amounts from {min(amounts):.2f} to {max(amounts):.2f} EUR "
        f"(auto-post cutoff: {classifier.confidence_cutoff:.0%})",
        title="[bold cyan]Counterfactual Price Sweep[/bold cyan]",
        border_style="cyan",
    ))

    table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("EUR", justify="right", no_wrap=True)
    table.add_column("Predicted GL", no_wrap=True)
    table.add_column("OPEX", justify="right", no_wrap=True)
    table.add_column("CAPEX", justify="right", no_wrap=True)
    table.add_column("Bar", style="blue", no_wrap=True)
    table.add_column("Conf.", justify="right", no_wrap=True)
    table.add_column("Routing", no_wrap=True, min_width=9)

    opex_key = ID2LABEL[LABEL2ID["GL_6220_IT_PERIPHERALS_OPEX"]]
    capex_key = ID2LABEL[LABEL2ID["GL_1400_IT_HARDWARE_CAPEX"]]

    prev_gl = None
    flip_amounts = []
    review_amounts = []

    for amt in amounts:
        res = classifier.predict(vendor, description, department, amt)
        gl = res['gl_account']
        conf = res['confidence']
        p_opex = res['all_predictions'].get(opex_key, 0.0)
        p_capex = res['all_predictions'].get(capex_key, 0.0)
        capex_ratio = p_capex / max(p_opex + p_capex, 1e-9)

        if prev_gl is not None and gl != prev_gl:
            flip_amounts.append(amt)
        prev_gl = gl

        if res['routing'] == "AUTO_POST":
            routing_cell = "[green]AUTO-POST[/green]"
        else:
            routing_cell = "[yellow]REVIEW[/yellow]"
            review_amounts.append(amt)

        gl_short = gl.replace("GL_", "")
        if gl == opex_key:
            gl_cell = f"[cyan]{gl_short}[/cyan]"
        elif gl == capex_key:
            gl_cell = f"[bold magenta]{gl_short}[/bold magenta]"
        else:
            gl_cell = gl_short

        filled = int(round(capex_ratio * 4))
        bar = "█" * filled + "░" * (4 - filled)

        table.add_row(
            f"{amt:,.2f}",
            gl_cell,
            f"{p_opex:.1%}",
            f"{p_capex:.1%}",
            bar,
            f"{conf:.1%}",
            routing_cell,
        )

    console.print(table)

    if flip_amounts:
        flips_str = ", ".join(f"{a:.2f} EUR" for a in flip_amounts)
        edges = classifier.band_edges or list(price_band_edges())
        rule_str = ", ".join(f"{e:.2f} EUR" for e in edges) if edges else "n/a"
        console.print(f"\n[bold]Decision flip observed at:[/bold] {flips_str} (accounting rule boundary: {rule_str})")
    if review_amounts:
        rev_str = ", ".join(f"{a:.2f}" for a in review_amounts)
        console.print(f"[bold]Routed to manual review (< {classifier.confidence_cutoff:.0%}):[/bold] {rev_str} EUR\n")
    else:
        console.print()


def interactive_mode(classifier: ERPClassifier, quick: bool = False):
    """Interactive mode - prompt for inputs."""
    console.print(Panel(
        "[bold cyan]ERP GL Account Classifier[/bold cyan]\n"
        "Interactive Mode - Enter transaction details\n"
        "Type 'quit' or 'exit' to stop",
        border_style="cyan",
    ))

    if not quick:
        console.print("\n[bold]Available departments:[/bold]")
        depts = [d for d in classifier.dept_vocab.keys() if d != "UNK"]
        console.print(f"  {', '.join(depts)}\n")

    while True:
        console.print("\n" + "─" * 60)

        vendor = Prompt.ask("[bold]Vendor[/bold]", default="Amazon Services Europe")
        if vendor.lower() in ['quit', 'exit']:
            break

        description = Prompt.ask("[bold]Description[/bold]")
        if description.lower() in ['quit', 'exit']:
            break

        department = Prompt.ask(
            "[bold]Department[/bold]",
            default="IT_OPS",
            choices=list(classifier.dept_vocab.keys()) if not quick else None,
        )
        if department.lower() in ['quit', 'exit']:
            break

        try:
            net_amount = FloatPrompt.ask("[bold]Amount (EUR)[/bold]")
        except (ValueError, KeyboardInterrupt):
            break

        console.print()
        with console.status("[bold green]Classifying..."):
            result = classifier.predict(vendor, description, department, net_amount)

        display_prediction(result, show_all=(not quick))

    console.print("\n[bold cyan]Goodbye![/bold cyan]")


def one_shot_mode(
    classifier: ERPClassifier,
    vendor: str,
    description: str,
    department: str,
    amount: float,
    show_all: bool = False,
):
    """One-shot mode - single prediction."""
    console.print("\n[bold]Transaction:[/bold]")
    console.print(f"  Vendor:      {vendor}")
    console.print(f"  Description: {description}")
    console.print(f"  Department:  {department}")
    console.print(f"  Amount:      {amount:.2f} EUR\n")

    with console.status("[bold green]Classifying..."):
        result = classifier.predict(vendor, description, department, amount)

    display_prediction(result, show_all=show_all)
    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="ERP GL Account Classifier - Command Line Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python classify_erp.py

  # Counterfactual sweep across the 800 EUR boundary
  python classify_erp.py --sweep 650:950:25

  # One-shot prediction
  python classify_erp.py \\
    --vendor "Amazon Services Europe" \\
    --description "Dell UltraSharp Monitor" \\
    --department IT_OPS \\
    --amount 450

  # Show all prediction probabilities and FiLM diagnostics
  python classify_erp.py \\
    --vendor "Amazon" \\
    --description "Coffee Beans" \\
    --department FACILITIES \\
    --amount 25 \\
    --show-all
        """,
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Path to trained model directory (default: models/multimodal-classifier)",
    )
    parser.add_argument(
        "--vendor",
        type=str,
        help="Vendor name (for one-shot or sweep mode)",
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Item description (for one-shot or sweep mode)",
    )
    parser.add_argument(
        "--department",
        type=str,
        help="Department (ENG, FACILITIES, HR, IT_OPS, MARKETING)",
    )
    parser.add_argument(
        "--amount",
        type=float,
        help="Net amount in EUR (for one-shot mode)",
    )
    parser.add_argument(
        "--sweep",
        nargs="?",
        const="650:950:25",
        default=None,
        help=(
            "Sweep amounts in EUR holding text and department fixed "
            "(format: 'start:stop:step' or 'a,b,c'; default when flag has no value: 650:950:25)"
        ),
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_CONFIDENCE_CUTOFF,
        help="Confidence threshold for AUTO-POST vs MANUAL REVIEW (default: 0.90)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick interactive mode (minimal prompts)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all prediction probabilities and FiLM modulation stats",
    )

    args = parser.parse_args()

    if not (0.0 <= args.cutoff <= 1.0):
        console.print(f"[red]Error: --cutoff must be between 0.0 and 1.0, got {args.cutoff}[/red]")
        sys.exit(1)

    try:
        classifier = ERPClassifier(args.model_dir, confidence_cutoff=args.cutoff)
    except ModelLoadError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        sys.exit(1)

    if args.sweep is not None:
        try:
            amounts = parse_sweep_spec(args.sweep)
        except ValueError as exc:
            console.print(f"[red]Error: {exc}[/red]")
            sys.exit(1)
        vendor = args.vendor or "Amazon Services Europe"
        description = args.description or "Dell UltraSharp U2723DE 27 Monitor"
        department = args.department or "IT_OPS"
        sweep_mode(classifier, vendor, description, department, amounts)
        return

    one_shot_args = [args.vendor, args.description, args.department, args.amount]

    if all(x is not None for x in one_shot_args):
        one_shot_mode(
            classifier,
            args.vendor,
            args.description,
            args.department,
            args.amount,
            args.show_all,
        )
    elif any(x is not None for x in one_shot_args):
        console.print(
            "[red]Error: For one-shot mode, provide all of: "
            "--vendor, --description, --department, --amount (or use --sweep)[/red]"
        )
        sys.exit(1)
    else:
        interactive_mode(classifier, args.quick)


if __name__ == "__main__":
    main()
