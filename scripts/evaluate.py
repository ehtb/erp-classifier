"""
Evaluate multimodal ERP classifier on test set.

Usage:
    python evaluate_multimodal.py \
        --model-dir models/multimodal-classifier \
        --test-data output/multimodal/test_multimodal.jsonl \
        --output-dir output/evaluation-multimodal
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

from transformers import AutoTokenizer, AutoConfig
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table

from src.model import ERPClassifierModel
from src.config import ID2LABEL, LABEL2ID

console = Console()


def load_model(model_dir: Path):
    """Load trained multimodal model."""
    console.print(f"🤖 Loading model from {model_dir}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load config
    config = AutoConfig.from_pretrained(model_dir)

    # Load model
    # Get number of departments from vocab file
    # (Assumes multimodal data was built in output/multimodal/)
    dept_vocab_path = Path("output/multimodal/dept_vocab.json")

    with open(dept_vocab_path) as f:
        dept_vocab = json.load(f)
    num_depts = len(dept_vocab['dept2id'])

    model = ERPClassifierModel(
        config=config,
        model_name="distilbert-base-uncased",
        num_depts=num_depts,
        num_continuous=2,
    )

    # Load weights
    model_path = model_dir / "pytorch_model.bin"
    if not model_path.exists():
        # Try safetensors
        model_path = model_dir / "model.safetensors"

    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    console.print(f"✓ Model loaded")
    return model, tokenizer


def evaluate(model, tokenizer, test_data: List[Dict], output_dir: Path):
    """Evaluate model on test set."""
    console.print(f"\n📊 Evaluating on {len(test_data)} test samples...")

    predictions = []
    true_labels = []
    confidences = []

    with torch.no_grad():
        for record in test_data:
            # Tokenize text
            encoding = tokenizer(
                record['text'],
                truncation=True,
                max_length=128,
                padding='max_length',
                return_tensors='pt',
            )

            # Prepare inputs
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            num_features = torch.tensor([record['num_features']], dtype=torch.float32)
            dept_id = torch.tensor([record['dept_id']], dtype=torch.long)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_features=num_features,
                dept_id=dept_id,
            )

            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            conf = probs[0, pred_idx].item()

            predictions.append(pred_idx)
            true_labels.append(record['label'])
            confidences.append(conf)

    # Convert to label strings
    pred_labels = [ID2LABEL[p] for p in predictions]
    true_labels_str = [ID2LABEL[t] for t in true_labels]

    # Compute metrics
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(
        true_labels_str,
        pred_labels,
        output_dict=True,
        zero_division=0,
    )

    # Print results
    console.print(f"\n[bold]Classification Report:[/bold]")
    console.print(f"  Accuracy: {accuracy:.4f}")
    console.print(f"  Weighted F1: {report['weighted avg']['f1-score']:.4f}")

    # Per-class table
    table = Table(title="\nPer-Class Metrics", show_header=True)
    table.add_column("GL Account", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Support", justify="right")

    for label in sorted(set(true_labels_str)):
        if label in report:
            metrics = report[label]
            table.add_row(
                label,
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                str(int(metrics['support'])),
            )

    console.print(table)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    labels = sorted(set(true_labels))
    label_names = [ID2LABEL[l] for l in labels]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[l.replace('GL_', '') for l in label_names],
        yticklabels=[l.replace('GL_', '') for l in label_names],
    )
    plt.title('Confusion Matrix - Multimodal ERP Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    console.print(f"\n💾 Saved confusion matrix to {cm_path}")

    # Analyze threshold classification (GL_6220 vs GL_1400)
    console.print(f"\n[bold]800 EUR Threshold Analysis:[/bold]")
    threshold_errors = analyze_threshold_classification(test_data, predictions, true_labels)

    # Save results
    results = {
        'accuracy': accuracy,
        'weighted_f1': report['weighted avg']['f1-score'],
        'classification_report': report,
        'threshold_analysis': threshold_errors,
    }

    results_path = output_dir / "evaluation_summary.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"\n💾 Saved evaluation summary to {results_path}")

    return results


def analyze_threshold_classification(test_data, predictions, true_labels):
    """Analyze how well the model learned the 800 EUR threshold."""
    gl_6220_idx = LABEL2ID['GL_6220_IT_PERIPHERALS_OPEX']  # < 800
    gl_1400_idx = LABEL2ID['GL_1400_IT_HARDWARE_CAPEX']     # >= 800

    threshold_cases = []
    for record, pred, true in zip(test_data, predictions, true_labels):
        if true in [gl_6220_idx, gl_1400_idx]:
            amount = record['metadata'].get('net_amount_eur', 0)
            threshold_cases.append({
                'amount': amount,
                'true_label': ID2LABEL[true],
                'pred_label': ID2LABEL[pred],
                'correct': pred == true,
            })

    if not threshold_cases:
        console.print("  No threshold cases in test set")
        return {}

    # Sort by amount
    threshold_cases.sort(key=lambda x: x['amount'])

    # Print examples around 800 EUR
    console.print("\n  Samples near 800 EUR threshold:")
    for case in threshold_cases:
        amount = case['amount']
        if 600 <= amount <= 1000:
            status = "✓" if case['correct'] else "✗"
            console.print(
                f"    {status} {amount:7.2f} EUR → "
                f"Predicted: {case['pred_label'].replace('GL_', '')} "
                f"(True: {case['true_label'].replace('GL_', '')})"
            )

    # Calculate accuracy for each side of threshold
    below_800 = [c for c in threshold_cases if c['amount'] < 800]
    above_800 = [c for c in threshold_cases if c['amount'] >= 800]

    below_acc = sum(1 for c in below_800 if c['correct']) / len(below_800) if below_800 else 0
    above_acc = sum(1 for c in above_800 if c['correct']) / len(above_800) if above_800 else 0

    console.print(f"\n  < 800 EUR (GL_6220): {below_acc*100:.1f}% accuracy ({sum(1 for c in below_800 if c['correct'])}/{len(below_800)})")
    console.print(f"  >= 800 EUR (GL_1400): {above_acc*100:.1f}% accuracy ({sum(1 for c in above_800 if c['correct'])}/{len(above_800)})")

    return {
        'below_800_accuracy': below_acc,
        'above_800_accuracy': above_acc,
        'below_800_count': len(below_800),
        'above_800_count': len(above_800),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate multimodal ERP classifier")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output/evaluation-multimodal")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model(Path(args.model_dir))

    # Load test data
    console.print(f"\n📂 Loading test data from {args.test_data}...")
    test_data = []
    with open(args.test_data) as f:
        for line in f:
            record = json.loads(line)
            record['num_features'] = np.array(record['num_features'], dtype=np.float32)
            test_data.append(record)

    console.print(f"✓ Loaded {len(test_data)} test samples")

    # Evaluate
    results = evaluate(model, tokenizer, test_data, output_dir)

    console.print(f"\n[bold green]✓ Evaluation complete![/bold green]")
    console.print(f"  Final test accuracy: {results['accuracy']:.1%}")
    console.print(f"  Final test F1: {results['weighted_f1']:.1%}")


if __name__ == "__main__":
    main()
