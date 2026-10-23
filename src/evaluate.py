"""
Evaluation script with error analysis and confidence calibration.

Provides:
1. Confusion matrix (8x8)
2. Per-class metrics
3. Misclassified samples with confidence scores
4. Confidence-based accuracy/coverage curves for production routing
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from transformers import AutoTokenizer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

from src.model import ERPClassifierModel
from src.config import NUM_LABELS, ID2LABEL, LABEL2ID

console = Console()


class ERPEvalDataset(torch.utils.data.Dataset):
    """Minimal evaluation dataset."""

    def __init__(self, jsonl_path: Path, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []

        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line)
                record["num_features"] = np.array(record["num_features"], dtype=np.float32)
                self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        encoding = self.tokenizer(
            record["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "num_features": torch.tensor(record["num_features"], dtype=torch.float32),
            "dept_id": torch.tensor(record["dept_id"], dtype=torch.long),
            "labels": torch.tensor(record["label"], dtype=torch.long),
            "text": record["text"],
            "net_amount": record.get("net_amount", 0.0),
            "dept_name": record.get("dept_name", "UNKNOWN"),
        }


def collate_fn(batch):
    """Custom collator preserving metadata."""
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "num_features": torch.stack([x["num_features"] for x in batch]),
        "dept_id": torch.stack([x["dept_id"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "text": [x["text"] for x in batch],
        "net_amount": [x["net_amount"] for x in batch],
        "dept_name": [x["dept_name"] for x in batch],
    }


def evaluate_model(
    model: ERPClassifierModel,
    dataloader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Run inference and collect predictions, confidences, and metadata.

    Returns:
        (true_labels, pred_labels, confidences, samples)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_confs = []
    all_samples = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            num_features = batch["num_features"].to(device)
            dept_id = batch["dept_id"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_features=num_features,
                dept_id=dept_id,
                return_dict=True,
            )

            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

            # Store samples with metadata
            for i in range(len(labels)):
                all_samples.append(
                    {
                        "text": batch["text"][i],
                        "net_amount": batch["net_amount"][i],
                        "dept_name": batch["dept_name"][i],
                        "true_label": int(labels[i].cpu()),
                        "pred_label": int(preds[i].cpu()),
                        "confidence": float(confs[i].cpu()),
                    }
                )

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_confs),
        all_samples,
    )


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """Print 8x8 confusion matrix with rich table."""
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_LABELS))

    table = Table(title="Confusion Matrix (Rows=True, Cols=Pred)")
    table.add_column("True \\ Pred", style="cyan")
    for i in range(NUM_LABELS):
        short_label = ID2LABEL[i].replace("GL_", "").replace("_", " ")[:12]
        table.add_column(short_label, justify="right")

    for i in range(NUM_LABELS):
        row_label = ID2LABEL[i].replace("GL_", "").replace("_", " ")[:12]
        row = [row_label] + [str(cm[i, j]) if cm[i, j] > 0 else "·" for j in range(NUM_LABELS)]
        table.add_row(*row)

    console.print(table)


def analyze_errors(samples: List[Dict]):
    """Print all misclassified samples."""
    errors = [s for s in samples if s["true_label"] != s["pred_label"]]

    if not errors:
        console.print("\n[green]✓ No errors! Perfect classification.[/green]")
        return

    console.print(f"\n[yellow]Misclassified Samples: {len(errors)}[/yellow]\n")

    table = Table()
    table.add_column("True", style="green")
    table.add_column("Pred", style="red")
    table.add_column("Conf", justify="right")
    table.add_column("Amount", justify="right")
    table.add_column("Dept", style="cyan")
    table.add_column("Text", overflow="fold", max_width=50)

    for err in errors:
        table.add_row(
            ID2LABEL[err["true_label"]].replace("GL_", "")[:20],
            ID2LABEL[err["pred_label"]].replace("GL_", "")[:20],
            f"{err['confidence']:.3f}",
            f"€{err['net_amount']:.0f}",
            err["dept_name"][:15],
            err["text"][:80],
        )

    console.print(table)


def compute_confidence_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    thresholds: List[float] = [0.50, 0.70, 0.80, 0.90, 0.95],
) -> Dict:
    """
    Compute accuracy and coverage at different confidence thresholds.

    For production ERP automation: samples below threshold route to MANUAL_REVIEW.
    """
    results = {}

    for thresh in thresholds:
        mask = confidences >= thresh
        coverage = mask.sum() / len(y_true)

        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
        else:
            acc = 0.0

        results[thresh] = {"accuracy": acc, "coverage": coverage, "count": int(mask.sum())}

    return results


def print_confidence_analysis(conf_results: Dict):
    """Print confidence-based routing table."""
    table = Table(title="Confidence-Based Routing (Auto-Post vs Manual Review)")
    table.add_column("Threshold", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Count", justify="right")
    table.add_column("Recommendation", style="bold")

    for thresh, res in sorted(conf_results.items()):
        rec = "✓ Auto-post" if res["accuracy"] >= 0.98 else "⚠ Review needed"
        color = "green" if res["accuracy"] >= 0.98 else "yellow"

        table.add_row(
            f"{thresh:.2f}",
            f"[{color}]{res['accuracy']:.4f}[/{color}]",
            f"{res['coverage']:.2%}",
            str(res["count"]),
            rec,
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, help="JSONL data file")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")

    args = parser.parse_args()

    console.print(f"\n[bold]Evaluating ERP Classifier[/bold]")
    console.print(f"  Model: {args.model_dir}")
    console.print(f"  Data: {args.data}")
    console.print(f"  Device: {args.device}")

    # Load model and tokenizer
    model = ERPClassifierModel.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    model.to(args.device)
    model.eval()

    # Load data
    dataset = ERPEvalDataset(Path(args.data), tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    console.print(f"  Samples: {len(dataset)}\n")

    # Run evaluation
    y_true, y_pred, confidences, samples = evaluate_model(model, dataloader, args.device)

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    console.print(f"[bold green]Overall Metrics[/bold green]")
    console.print(f"  Accuracy: {acc:.4f}")
    console.print(f"  Macro F1: {macro_f1:.4f}")
    console.print(f"  Macro Precision: {macro_p:.4f}")
    console.print(f"  Macro Recall: {macro_r:.4f}\n")

    # Per-class report
    console.print("[bold]Per-Class Metrics[/bold]")
    report = classification_report(y_true, y_pred, target_names=[ID2LABEL[i] for i in range(NUM_LABELS)])
    console.print(report)

    # Confusion matrix
    print_confusion_matrix(y_true, y_pred)

    # Error analysis
    analyze_errors(samples)

    # Confidence calibration
    console.print("\n[bold cyan]Confidence Calibration for Production Routing[/bold cyan]")
    conf_results = compute_confidence_curve(y_true, y_pred, confidences)
    print_confidence_analysis(conf_results)

    console.print(f"\n[green]✓ Evaluation complete![/green]")


if __name__ == "__main__":
    main()
