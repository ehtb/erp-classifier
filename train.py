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
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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


def compute_metrics(pred: EvalPrediction) -> Dict:
    """Compute accuracy, precision, recall, F1."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


def main():
    parser = argparse.ArgumentParser(description="Train multimodal ERP classifier")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="models/multimodal-classifier")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
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

    # Count departments
    train_records = train_dataset.records
    num_depts = max(r['dept_id'] for r in train_records) + 1
    console.print(f"  Departments: {num_depts} unique values")

    # Create model
    console.print(f"\n🤖 Creating multimodal model...")
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = NUM_LABELS

    model = ERPClassifierModel(
        config=config,
        model_name=args.model_name,
        num_depts=num_depts,
        num_continuous=2,  # net_amount, log1p(net_amount)
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

    # Calculate warmup steps (10% of total training steps)
    total_steps = (len(train_dataset) // args.batch_size) * args.num_epochs
    warmup_steps = int(total_steps * 0.1)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        report_to="none",
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=MultimodalDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
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
