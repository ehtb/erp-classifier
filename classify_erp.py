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

    # Quick mode (minimal prompts)
    python classify_erp.py --quick
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from transformers import AutoTokenizer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, FloatPrompt

from src.model import ERPClassifierModel
from src.config import ID2LABEL
from src.text import scrub_prices

console = Console()


class ERPClassifier:
    """Classifier with loaded model and artifacts."""

    def __init__(self, model_dir: str = "models/multimodal-v2"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.tokenizer = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.dept_vocab = None
        self.id2dept = None

        self._load_model()

    def _artifact(self, name: str) -> Path:
        """Preprocessing artifact saved next to the checkpoint.

        No fallback: mixing the scaler/vocabulary of a different dataset build with
        this checkpoint silently produces wrong probabilities.
        """
        path = self.model_dir / name
        if not path.exists():
            console.print(f"[red]Error: required artifact not found: {path}[/red]")
            console.print(
                "[yellow]The checkpoint must ship with its own scaler/vocabulary. "
                "Retrain with train.py, which copies them next to the weights.[/yellow]"
            )
            sys.exit(1)
        return path

    def _load_model(self):
        """Load model, tokenizer, and auxiliary files."""
        console.print(f"\n[bold cyan]Loading ERP Classifier...[/bold cyan]")

        if not self.model_dir.exists():
            console.print(f"[red]Error: Model directory not found: {self.model_dir}[/red]")
            console.print(f"[yellow]Available models:[/yellow]")
            models_dir = Path("models")
            if models_dir.exists():
                for model_path in models_dir.iterdir():
                    if model_path.is_dir():
                        console.print(f"  - {model_path}")
            console.print(f"\n[yellow]Or train a new model:[/yellow] python train.py --output-dir {self.model_dir}")
            sys.exit(1)

        # Load tokenizer
        with console.status("[bold green]Loading tokenizer..."):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        # Load scaler
        scaler_path = self._artifact("scaler.json")

        with open(scaler_path) as f:
            scaler_params = json.load(f)
            self.scaler_mean = np.array(scaler_params['mean'])
            self.scaler_scale = np.array(scaler_params['scale'])

        # Load dept vocab
        vocab_path = self._artifact("dept_vocab.json")

        with open(vocab_path) as f:
            vocab_data = json.load(f)
            self.dept_vocab = vocab_data['dept2id']
            self.id2dept = {int(k): v for k, v in vocab_data['id2dept'].items()}

        # Load model using from_pretrained (handles architecture automatically)
        with console.status("[bold green]Loading model weights..."):
            self.model = ERPClassifierModel.from_pretrained(self.model_dir)
            self.model.eval()

        if self.model.num_labels != len(ID2LABEL):
            console.print(
                f"[red]Error: checkpoint has {self.model.num_labels} labels, "
                f"but the code expects {len(ID2LABEL)}.[/red]"
            )
            sys.exit(1)

        console.print(f"[bold green]✓ Model loaded successfully![/bold green]\n")

    def predict(
        self,
        vendor: str,
        description: str,
        department: str,
        net_amount_eur: float
    ) -> Dict:
        """
        Predict GL account for a transaction.

        Returns:
            dict with keys: gl_account, confidence, all_predictions
        """
        # Prepare text input. Scrub explicit price quotes: training text is scrubbed,
        # and the amount belongs in the numeric branch, not in the description.
        text = f"[VENDOR] {vendor} [DESC] {scrub_prices(description)}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding='max_length',
            return_tensors='pt',
        )

        # Prepare numerical features (clamp at 0 so negative credit-note amounts do not NaN in log1p)
        safe_amount = max(0.0, float(net_amount_eur))
        num_features = np.array([net_amount_eur, np.log1p(safe_amount)])
        num_features = (num_features - self.scaler_mean) / self.scaler_scale

        # Prepare department ID
        dept_id = self.dept_vocab.get(department.upper(), 0)  # 0 = UNK

        # Inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                num_features=torch.from_numpy(num_features).unsqueeze(0).float(),
                dept_id=torch.tensor([dept_id], dtype=torch.long),
            )

            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[0, pred_idx].item()

            # Get all predictions
            all_predictions = {
                ID2LABEL[i]: float(probs[0, i])
                for i in range(len(ID2LABEL))
            }

        return {
            'gl_account': ID2LABEL[pred_idx],
            'confidence': confidence,
            'all_predictions': all_predictions,
        }


def display_prediction(result: Dict, show_all: bool = False):
    """Display prediction results in a nice format."""

    # Main prediction panel
    gl_account = result['gl_account']
    confidence = result['confidence']

    # Color based on confidence
    if confidence > 0.9:
        color = "green"
    elif confidence > 0.7:
        color = "yellow"
    else:
        color = "red"

    console.print(Panel(
        f"[bold {color}]{gl_account}[/bold {color}]\n"
        f"Confidence: {confidence:.1%}",
        title="[bold]Prediction[/bold]",
        border_style=color
    ))

    # Show all predictions if requested
    if show_all or confidence < 0.7:
        console.print("\n[bold]All Predictions:[/bold]")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("GL Account", style="cyan")
        table.add_column("Probability", justify="right")
        table.add_column("Bar", style="blue")

        # Sort by probability
        sorted_preds = sorted(
            result['all_predictions'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for gl, prob in sorted_preds[:5]:  # Top 5
            bar = "█" * int(prob * 20)
            table.add_row(
                gl.replace("GL_", ""),
                f"{prob:.1%}",
                bar
            )

        console.print(table)


def interactive_mode(classifier: ERPClassifier, quick: bool = False):
    """Interactive mode - prompt for inputs."""

    console.print(Panel(
        "[bold cyan]ERP GL Account Classifier[/bold cyan]\n"
        "Interactive Mode - Enter transaction details\n"
        "Type 'quit' or 'exit' to stop",
        border_style="cyan"
    ))

    # Show available departments
    if not quick:
        console.print("\n[bold]Available departments:[/bold]")
        depts = [d for d in classifier.dept_vocab.keys() if d != "UNK"]
        console.print(f"  {', '.join(depts)}\n")

    while True:
        console.print("\n" + "─" * 60)

        # Get inputs
        vendor = Prompt.ask("[bold]Vendor[/bold]", default="Amazon Services Europe")
        if vendor.lower() in ['quit', 'exit']:
            break

        description = Prompt.ask("[bold]Description[/bold]")
        if description.lower() in ['quit', 'exit']:
            break

        department = Prompt.ask(
            "[bold]Department[/bold]",
            default="IT_OPS",
            choices=list(classifier.dept_vocab.keys()) if not quick else None
        )
        if department.lower() in ['quit', 'exit']:
            break

        try:
            net_amount = FloatPrompt.ask("[bold]Amount (EUR)[/bold]")
        except (ValueError, KeyboardInterrupt):
            break

        # Predict
        console.print()
        with console.status("[bold green]Classifying..."):
            result = classifier.predict(vendor, description, department, net_amount)

        # Display
        display_prediction(result, show_all=(not quick))

    console.print("\n[bold cyan]Goodbye![/bold cyan]")


def one_shot_mode(
    classifier: ERPClassifier,
    vendor: str,
    description: str,
    department: str,
    amount: float,
    show_all: bool = False
):
    """One-shot mode - single prediction."""

    # Display inputs
    console.print("\n[bold]Transaction:[/bold]")
    console.print(f"  Vendor:      {vendor}")
    console.print(f"  Description: {description}")
    console.print(f"  Department:  {department}")
    console.print(f"  Amount:      {amount:.2f} EUR\n")

    # Predict
    with console.status("[bold green]Classifying..."):
        result = classifier.predict(vendor, description, department, amount)

    # Display
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

  # Quick interactive mode (less verbose)
  python classify_erp.py --quick

  # One-shot prediction
  python classify_erp.py \\
    --vendor "Amazon Services" \\
    --description "Dell UltraSharp Monitor" \\
    --department IT_OPS \\
    --amount 450

  # Show all prediction probabilities
  python classify_erp.py \\
    --vendor "Amazon" \\
    --description "Coffee Beans" \\
    --department FACILITIES \\
    --amount 25 \\
    --show-all
        """
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/multimodal-v2",
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--vendor",
        type=str,
        help="Vendor name (for one-shot mode)"
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Item description (for one-shot mode)"
    )
    parser.add_argument(
        "--department",
        type=str,
        help="Department (ENG, FACILITIES, HR, IT_OPS, MARKETING)"
    )
    parser.add_argument(
        "--amount",
        type=float,
        help="Net amount in EUR (for one-shot mode)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick interactive mode (minimal prompts)"
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all prediction probabilities"
    )

    args = parser.parse_args()

    # Load classifier
    classifier = ERPClassifier(args.model_dir)

    # Determine mode
    one_shot_args = [args.vendor, args.description, args.department, args.amount]

    if all(x is not None for x in one_shot_args):
        # One-shot mode
        one_shot_mode(
            classifier,
            args.vendor,
            args.description,
            args.department,
            args.amount,
            args.show_all
        )
    elif any(x is not None for x in one_shot_args):
        # Partial args provided
        console.print("[red]Error: For one-shot mode, provide all of: --vendor, --description, --department, --amount[/red]")
        sys.exit(1)
    else:
        # Interactive mode
        interactive_mode(classifier, args.quick)


if __name__ == "__main__":
    main()
