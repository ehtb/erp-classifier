"""
Evaluation script for ERP GL account classifier.

Provides:
1. Classification report
2. Confusion matrix
3. Counterfactual accuracy (price flips, department flips)
4. Single-sample inference latency
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table

from .config import LABEL2ID, ID2LABEL, NUM_LABELS

console = Console()



class GLClassifier:
    """Wrapper for trained GL account classifier."""

    def __init__(self, model_path: Path, device: str = None):
        """
        Initialize classifier.

        Args:
            model_path: Path to saved model directory
            device: Device to use (cuda/cpu, auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        console.print(f"🤖 Loading model from {model_path}")
        console.print(f"   Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True  # For LoRA adapters
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict GL account for a single text.

        Returns:
            (predicted_label, confidence, all_probabilities)
        """
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_idx].item()

        predicted_label = ID2LABEL[pred_idx]

        # Get all probabilities
        all_probs = {
            ID2LABEL[i]: probs[0, i].item()
            for i in range(NUM_LABELS)
        }

        return predicted_label, confidence, all_probs

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Tuple[str, float]]:
        """
        Predict GL accounts for a batch of texts.

        Returns:
            List of (predicted_label, confidence) tuples
        """
        predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoding = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt',
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

            pred_indices = torch.argmax(probs, dim=-1).cpu().numpy()
            confidences = probs[range(len(batch_texts)), pred_indices].cpu().numpy()

            for pred_idx, conf in zip(pred_indices, confidences):
                predictions.append((ID2LABEL[pred_idx], float(conf)))

        return predictions

    def measure_latency(self, text: str, num_runs: int = 100) -> Dict[str, float]:
        """Measure single-sample inference latency."""
        # Warmup
        for _ in range(10):
            self.predict(text)

        # Measure
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.predict(text)
            latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms

        return {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
        }


def evaluate_standard(
    classifier: GLClassifier,
    test_records: List[dict],
    output_dir: Path,
) -> Dict:
    """
    Evaluate on standard test set.

    Returns metrics dict.
    """
    console.print(f"\n📊 Evaluating on {len(test_records)} test records...")

    # Predict
    texts = [r['text'] for r in test_records]
    # Convert integer labels to strings (HuggingFace datasets store labels as ClassLabel ints)
    true_labels = [ID2LABEL[r['label']] if isinstance(r['label'], int) else r['label']
                   for r in test_records]

    predictions = classifier.predict_batch(texts)
    pred_labels = [p[0] for p in predictions]
    confidences = [p[1] for p in predictions]

    # Compute metrics
    accuracy = accuracy_score(true_labels, pred_labels)

    # Classification report
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=sorted(set(true_labels)),
        output_dict=True,
        zero_division=0,
    )

    console.print(f"\n[bold cyan]Classification Report:[/bold cyan]")
    console.print(f"  Accuracy: {accuracy:.4f}")
    console.print(f"  Weighted F1: {report['weighted avg']['f1-score']:.4f}")

    # Print per-class metrics
    table = Table(title="Per-Class Metrics", show_header=True)
    table.add_column("GL Account", style="cyan")
    table.add_column("Precision", justify="right", style="green")
    table.add_column("Recall", justify="right", style="yellow")
    table.add_column("F1", justify="right", style="magenta")
    table.add_column("Support", justify="right")

    for gl_account in sorted(set(true_labels)):
        if gl_account in report:
            metrics = report[gl_account]
            table.add_row(
                gl_account,
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                str(int(metrics['support'])),
            )

    console.print(table)

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=sorted(set(true_labels)))

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[gl.replace('GL_', '') for gl in sorted(set(true_labels))],
        yticklabels=[gl.replace('GL_', '') for gl in sorted(set(true_labels))],
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    console.print(f"\n💾 Saved confusion matrix to {cm_path}")

    # Save detailed results
    results_path = output_dir / "test_results.jsonl"
    with open(results_path, 'w') as f:
        for record, pred_label, conf in zip(test_records, pred_labels, confidences):
            result = {
                "text": record['text'],
                "true_label": record['label'],
                "predicted_label": pred_label,
                "confidence": conf,
                "correct": pred_label == record['label'],
                "metadata": record.get('metadata', {}),
            }
            f.write(json.dumps(result) + '\n')

    console.print(f"💾 Saved detailed results to {results_path}")

    return {
        "accuracy": accuracy,
        "weighted_f1": report['weighted avg']['f1-score'],
        "classification_report": report,
    }


def evaluate_counterfactuals(
    classifier: GLClassifier,
    counterfactual_records: List[dict],
    output_dir: Path,
) -> Dict:
    """
    Evaluate counterfactual robustness.

    Tests if model correctly handles:
    1. Price flips around 800 EUR threshold
    2. Department flips for same item
    """
    console.print(f"\n🔄 Evaluating {len(counterfactual_records)} counterfactual records...")

    # Group by pair_id
    pairs = defaultdict(list)
    for record in counterfactual_records:
        pair_id = record.get('pair_id')
        if pair_id is not None:
            pairs[pair_id].append(record)

    # Predict for all records
    texts = [r['text'] for r in counterfactual_records]
    predictions = classifier.predict_batch(texts)

    # Add predictions to records
    for record, (pred_label, conf) in zip(counterfactual_records, predictions):
        record['predicted_label'] = pred_label
        record['confidence'] = conf
        record['correct'] = pred_label == record['label']

    # Analyze pairs
    price_flip_correct = 0
    dept_flip_correct = 0
    total_price_flips = 0
    total_dept_flips = 0

    pair_results = []

    for pair_id, pair_records in pairs.items():
        if len(pair_records) != 2:
            continue

        low, high = pair_records[0], pair_records[1]
        if low.get('pair_position') == 'high':
            low, high = high, low

        # Determine flip type based on metadata
        low_meta = low.get('metadata', {})
        high_meta = high.get('metadata', {})

        is_price_flip = (
            low_meta.get('base_category') == 'IT_HARDWARE' and
            high_meta.get('base_category') == 'IT_HARDWARE' and
            low_meta.get('department') == high_meta.get('department')
        )

        is_dept_flip = (
            low_meta.get('department') != high_meta.get('department') and
            low_meta.get('net_amount_eur') == high_meta.get('net_amount_eur')
        )

        both_correct = low['correct'] and high['correct']

        pair_result = {
            'pair_id': pair_id,
            'is_price_flip': is_price_flip,
            'is_dept_flip': is_dept_flip,
            'both_correct': both_correct,
            'low': {
                'text': low['text'],
                'true_label': low['label'],
                'predicted_label': low['predicted_label'],
                'correct': low['correct'],
            },
            'high': {
                'text': high['text'],
                'true_label': high['label'],
                'predicted_label': high['predicted_label'],
                'correct': high['correct'],
            },
        }

        pair_results.append(pair_result)

        if is_price_flip:
            total_price_flips += 1
            if both_correct:
                price_flip_correct += 1

        if is_dept_flip:
            total_dept_flips += 1
            if both_correct:
                dept_flip_correct += 1

    # Print results
    console.print(f"\n[bold cyan]Counterfactual Results:[/bold cyan]")

    if total_price_flips > 0:
        price_flip_acc = price_flip_correct / total_price_flips
        console.print(f"  Price flip pairs (800 EUR threshold): {price_flip_correct}/{total_price_flips} ({price_flip_acc:.1%})")

    if total_dept_flips > 0:
        dept_flip_acc = dept_flip_correct / total_dept_flips
        console.print(f"  Department flip pairs: {dept_flip_correct}/{total_dept_flips} ({dept_flip_acc:.1%})")

    # Save results
    results_path = output_dir / "counterfactual_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'price_flip_accuracy': price_flip_correct / total_price_flips if total_price_flips > 0 else None,
            'dept_flip_accuracy': dept_flip_correct / total_dept_flips if total_dept_flips > 0 else None,
            'total_price_flips': total_price_flips,
            'total_dept_flips': total_dept_flips,
            'pair_results': pair_results,
        }, f, indent=2)

    console.print(f"💾 Saved counterfactual results to {results_path}")

    return {
        'price_flip_accuracy': price_flip_correct / total_price_flips if total_price_flips > 0 else None,
        'dept_flip_accuracy': dept_flip_correct / total_dept_flips if total_dept_flips > 0 else None,
    }


def main():
    """CLI entry point."""
    import argparse
    from datasets import load_from_disk

    parser = argparse.ArgumentParser(description="Evaluate GL account classifier")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory with trained model",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to HuggingFace dataset directory (must contain a 'test' split)",
    )
    parser.add_argument(
        "--counterfactuals",
        type=str,
        help="Path to HuggingFace counterfactual dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for evaluation results (defaults to model-dir/eval)",
    )
    parser.add_argument(
        "--measure-latency",
        action="store_true",
        help="Measure single-sample inference latency",
    )

    args = parser.parse_args()

    model_path = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_path / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load classifier
    classifier = GLClassifier(model_path)

    # Evaluate on test split
    test_records = []
    test_metrics = {}
    dataset_dict = load_from_disk(args.data_path)
    if "test" in dataset_dict:
        test_records = list(dataset_dict["test"])
        test_metrics = evaluate_standard(classifier, test_records, output_dir)
    else:
        console.print(f"[yellow]Warning: no 'test' split in {args.data_path}, skipping standard evaluation[/yellow]")

    # Evaluate on counterfactuals
    counterfactual_metrics = {}
    if args.counterfactuals:
        cf_dataset = load_from_disk(args.counterfactuals)
        # Counterfactuals are saved as a plain Dataset, not a DatasetDict
        cf_records = list(cf_dataset) if hasattr(cf_dataset, '__iter__') else []
        if cf_records:
            counterfactual_metrics = evaluate_counterfactuals(classifier, cf_records, output_dir)
        else:
            console.print(f"[yellow]Warning: counterfactual dataset at {args.counterfactuals} is empty[/yellow]")

    # Measure latency
    if args.measure_latency and test_records:
        console.print(f"\n⏱️  Measuring inference latency...")
        sample_text = test_records[0]['text']
        latency_stats = classifier.measure_latency(sample_text)

        console.print(f"\n[bold cyan]Latency Statistics (100 runs):[/bold cyan]")
        console.print(f"  Mean: {latency_stats['mean_ms']:.2f} ms")
        console.print(f"  Std:  {latency_stats['std_ms']:.2f} ms")
        console.print(f"  P50:  {latency_stats['p50_ms']:.2f} ms")
        console.print(f"  P95:  {latency_stats['p95_ms']:.2f} ms")
        console.print(f"  P99:  {latency_stats['p99_ms']:.2f} ms")

        latency_path = output_dir / "latency_stats.json"
        with open(latency_path, 'w') as f:
            json.dump(latency_stats, f, indent=2)

    # Save summary
    summary = {
        'test_metrics': test_metrics,
        'counterfactual_metrics': counterfactual_metrics,
    }

    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n💾 Saved evaluation summary to {summary_path}")
    console.print(f"\n[bold green]✓ Evaluation complete![/bold green]")


if __name__ == "__main__":
    main()
