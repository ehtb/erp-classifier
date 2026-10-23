"""
Train multimodal tabular-text ERP classifier.

This model learns the 800 EUR threshold natively via backpropagation,
NOT from hardcoded rules in the data.

Usage:
    python train_multimodal.py \
        --train-data output/multimodal/train_multimodal.jsonl \
        --val-data output/multimodal/val_multimodal.jsonl \
        --output-dir models/multimodal-classifier \
        --num-epochs 15
"""

import argparse
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from collections import Counter

from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from rich.console import Console

from src.model import ERPClassifierModel
from src.config import NUM_LABELS, ID2LABEL

console = Console()


class MultimodalERPDataset(Dataset):
    """PyTorch Dataset for multimodal ERP data."""

    def __init__(self, jsonl_path: Path, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line)
                record['num_features'] = np.array(record['num_features'], dtype=np.float32)
                self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # Tokenize text
        encoding = self.tokenizer(
            record['text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'num_features': torch.tensor(record['num_features'], dtype=torch.float32),
            'dept_id': torch.tensor(record['dept_id'], dtype=torch.long),
            'labels': torch.tensor(record['label'], dtype=torch.long),
        }


@dataclass
class MultimodalDataCollator:
    """
    Custom data collator for multimodal inputs.
    Handles text, numerical, and categorical features.
    """

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'num_features': torch.stack([f['num_features'] for f in features]),
            'dept_id': torch.stack([f['dept_id'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features]),
        }
        return batch


def compute_class_weights(train_records: List[Dict], num_labels: int) -> torch.Tensor:
    """
    Safely compute balanced class weights even if a class has low/zero support.
    """
    labels = [r['label'] for r in train_records]
    label_counts = Counter(labels)

    console.print("\n📊 Class distribution in Train:")
    for label_id in range(num_labels):
        label_name = ID2LABEL.get(label_id, f"Label_{label_id}")
        count = label_counts.get(label_id, 0)
        console.print(f"  [{label_id}] {label_name}: {count} samples")

    unique_classes = np.unique(labels)
    computed = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels,
    )

    weights = np.ones(num_labels, dtype=np.float32)
    for cls_id, w in zip(unique_classes, computed):
        # Cap extreme weights at 5.0 to prevent exploding gradients on tiny classes
        weights[cls_id] = min(float(w), 5.0)

    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(pred: EvalPrediction) -> Dict:
    """Compute accuracy, macro F1 (for model selection), and weighted F1."""
    labels = pred.label_ids
    preds = pred.predictions if isinstance(pred.predictions, np.ndarray) else pred.predictions[0]
    preds = preds.argmax(-1)

    acc = accuracy_score(labels, preds)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )

    return {
        'accuracy': acc,
        'macro_f1': m_f1,       # Use this for metric_for_best_model
        'weighted_f1': w_f1,
        'macro_precision': m_prec,
        'macro_recall': m_rec,
    }


def main():
    parser = argparse.ArgumentParser(description="Train multimodal ERP classifier")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="models/multimodal-classifier")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--tabular-lr", type=float, default=5e-4)
    parser.add_argument("--max-length", type=int, default=128)

    args = parser.parse_args()

    console.print("\n[bold]Training Multimodal ERP Classifier[/bold]")
    console.print(f"  Model: {args.model_name}")
    console.print(f"  Train data: {args.train_data}")
    console.print(f"  Val data: {args.val_data}")
    console.print(f"  Epochs: {args.num_epochs}")
    console.print(f"  Batch size: {args.batch_size}")
    console.print(f"  Learning rate: {args.learning_rate}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create datasets
    console.print("\n📂 Loading datasets...")
    train_dataset = MultimodalERPDataset(Path(args.train_data), tokenizer, args.max_length)
    val_dataset = MultimodalERPDataset(Path(args.val_data), tokenizer, args.max_length)

    console.print(f"  Train: {len(train_dataset)} samples")
    console.print(f"  Val: {len(val_dataset)} samples")

    # Safely count departments across both train and val (+2 buffer for test)
    train_records = train_dataset.records
    all_records = train_dataset.records + val_dataset.records
    num_depts = max(r['dept_id'] for r in all_records) + 2
    console.print(f"  Departments embedding size: {num_depts}")

    # Compute class weights to handle imbalance from Critic filter
    class_weights = compute_class_weights(train_records, NUM_LABELS)

    # Create model
    console.print(f"\n🤖 Creating multimodal model...")
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = NUM_LABELS

    model = ERPClassifierModel(
        config=config,
        model_name=args.model_name,
        num_depts=num_depts,
        num_continuous=2,  # z_score(net_amount), log1p(net_amount)
        class_weights=class_weights,
    )

    console.print(f"  Architecture:")
    console.print(f"    Text encoder: {args.model_name}")
    console.print(f"    Numerical branch: [2] → [64]")
    console.print(f"    Department branch: [{num_depts}] → [32]")
    console.print(f"    Fusion: [768 + 64 + 32] → [256] → [128] → [{NUM_LABELS}]")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"  Total params: {total_params:,}")
    console.print(f"  Trainable: {trainable_params:,}")

    # Training arguments
    output_dir = Path(args.output_dir)

    # Calculate warmup steps (15% of total training steps for smoother annealing)
    total_steps = (len(train_dataset) // args.batch_size) * args.num_epochs
    warmup_steps = int(total_steps * 0.15)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        remove_unused_columns=False,
    )

    # Differential learning rates with no weight decay on bias / LayerNorm
    def optimizer_factory(model):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "dept_norm.weight", "text_norm.weight"]

        encoder_named = list(model.encoder.named_parameters())
        encoder_ids = {id(p) for _, p in encoder_named}
        tabular_named = [
            (n, p) for n, p in model.named_parameters()
            if id(p) not in encoder_ids and p.requires_grad
        ]

        optimizer_grouped_parameters = [
            # 1. Encoder with weight decay
            {
                "params": [p for n, p in encoder_named if not any(nd in n for nd in no_decay)],
                "lr": args.learning_rate,
                "weight_decay": 0.01,
            },
            # 2. Encoder without weight decay (biases & LayerNorm)
            {
                "params": [p for n, p in encoder_named if any(nd in n for nd in no_decay)],
                "lr": args.learning_rate,
                "weight_decay": 0.0,
            },
            # 3. Tabular & Fusion heads with weight decay
            {
                "params": [p for n, p in tabular_named if not any(nd in n for nd in no_decay)],
                "lr": args.tabular_lr,
                "weight_decay": 0.01,
            },
            # 4. Tabular & Fusion heads without weight decay (critical for threshold biases!)
            {
                "params": [p for n, p in tabular_named if any(nd in n for nd in no_decay)],
                "lr": args.tabular_lr,
                "weight_decay": 0.0,
            },
        ]

        console.print(f"\n⚙️  Differential learning rates configured:")
        console.print(f"  Encoder (DistilBERT): {args.learning_rate}")
        console.print(f"  Tabular & Fusion heads: {args.tabular_lr} (0.0 weight decay on threshold biases)")

        return AdamW(optimizer_grouped_parameters)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=MultimodalDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        optimizers=(optimizer_factory(model), None),  # (optimizer, lr_scheduler)
    )

    # Train
    console.print(f"\n🚀 Starting training...\n")
    train_result = trainer.train()

    # Save
    console.print(f"\n💾 Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Final evaluation
    console.print(f"\n📊 Final validation metrics:")
    metrics = trainer.evaluate()
    for key, value in metrics.items():
        if key.startswith('eval_'):
            console.print(f"  {key.replace('eval_', '')}: {value:.4f}")

    console.print(f"\n[bold green]✓ Training complete![/bold green]")
    console.print(f"  Model saved to: {output_dir}")

    # Save training summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'model_name': args.model_name,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'final_metrics': metrics,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }, f, indent=2)

    console.print(f"  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
