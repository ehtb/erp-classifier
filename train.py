"""
Train multimodal tabular-text ERP classifier.

The threshold lives in the accounting rules (`src/config.py`). Training uses
rule-derived counterfactual amounts to distill that rule into the numeric branch;
it does not discover the boundary from scratch.

Usage:
    python train.py \
        --train-data output/multimodal/train_multimodal.jsonl \
        --val-data output/multimodal/val_multimodal.jsonl \
        --output-dir models/multimodal-classifier \
        --num-epochs 8
"""

import argparse
import json
import shutil
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
    set_seed,
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
        )

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'num_features': torch.tensor(record['num_features'], dtype=torch.float32),
            'dept_id': torch.tensor(record['dept_id'], dtype=torch.long),
            'labels': torch.tensor(record['label'], dtype=torch.long),
            'sample_weight': torch.tensor(record.get('sample_weight', 1.0), dtype=torch.float32),
        }


@dataclass
class MultimodalDataCollator:
    """
    Custom data collator for multimodal inputs.
    Handles text (dynamic padding to the longest sequence in the batch),
    numerical, and categorical features.
    """

    pad_token_id: int = 0

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f['input_ids']) for f in features)
        pad = lambda t, v: torch.nn.functional.pad(t, (0, max_len - len(t)), value=v)
        batch = {
            'input_ids': torch.stack([pad(f['input_ids'], self.pad_token_id) for f in features]),
            'attention_mask': torch.stack([pad(f['attention_mask'], 0) for f in features]),
            'num_features': torch.stack([f['num_features'] for f in features]),
            'dept_id': torch.stack([f['dept_id'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features]),
            'sample_weight': torch.stack([f['sample_weight'] for f in features]),
        }
        return batch


def describe_weights(train_records: List[Dict], num_labels: int):
    """Print raw counts and effective (sample-weighted) class mass."""
    raw = Counter(r['label'] for r in train_records)
    mass = Counter()
    for r in train_records:
        mass[r['label']] += float(r.get('sample_weight', 1.0))
    console.print("\n📊 Class distribution in Train (count / effective weight):")
    for label_id in range(num_labels):
        label_name = ID2LABEL.get(label_id, f"Label_{label_id}")
        console.print(
            f"  [{label_id}] {label_name}: {raw.get(label_id, 0)} / {mass.get(label_id, 0.0):.1f}"
        )
    return raw, mass


def compute_class_weights(train_records: List[Dict], num_labels: int) -> torch.Tensor:
    """Balanced class weights, capped at 5.0 so tiny classes do not explode."""
    labels = [r['label'] for r in train_records]
    unique_classes = np.unique(labels)
    computed = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels,
    )

    weights = np.ones(num_labels, dtype=np.float32)
    for cls_id, w in zip(unique_classes, computed):
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-encoding", choices=["ple", "periodic"], default="ple",
        help="Numeric branch encoding; 'ple' uses the bin edges in numeric_bins.json",
    )
    parser.add_argument(
        "--class-weight-mode", choices=["cf", "balanced", "none"], default="cf",
        help=(
            "'cf' (default) uses the per-sample weights written by the dataset builder "
            "(counterfactual copies count 1/(k+1)) to preserve the source class prior; "
            "'balanced' and 'none' are kept strictly for the class-weight ablation"
        ),
    )

    args = parser.parse_args()

    console.print("\n[bold]Training Multimodal ERP Classifier[/bold]")
    console.print(f"  Model: {args.model_name}")
    console.print(f"  Train data: {args.train_data}")
    console.print(f"  Val data: {args.val_data}")
    console.print(f"  Epochs: {args.num_epochs}")
    console.print(f"  Batch size: {args.batch_size}")
    console.print(f"  Learning rate: {args.learning_rate}")

    set_seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create datasets
    console.print("\n📂 Loading datasets...")
    train_dataset = MultimodalERPDataset(Path(args.train_data), tokenizer, args.max_length)
    val_dataset = MultimodalERPDataset(Path(args.val_data), tokenizer, args.max_length)

    console.print(f"  Train: {len(train_dataset)} samples")
    console.print(f"  Val: {len(val_dataset)} samples")

    # Department embedding size comes from the vocabulary saved by the dataset builder
    # (includes UNK at 0), the same source evaluation/inference use.
    train_records = train_dataset.records
    data_dir = Path(args.train_data).parent
    with open(data_dir / "dept_vocab.json") as f:
        num_depts = len(json.load(f)['dept2id'])
    console.print(f"  Departments embedding size: {num_depts}")

    # Weighting policy. 'cf' restores the source class prior via the per-sample
    # weights emitted by the dataset builder; the other modes overwrite them.
    describe_weights(train_records, NUM_LABELS)
    if args.class_weight_mode == "balanced":
        class_weights = compute_class_weights(train_records, NUM_LABELS)
        for r in train_records:
            r['sample_weight'] = 1.0
        use_class_weights = True
    elif args.class_weight_mode == "none":
        class_weights = None
        for r in train_records:
            r['sample_weight'] = 1.0
        use_class_weights = False
    else:  # cf
        class_weights = None
        use_class_weights = False
    console.print(f"  Class-weight mode: {args.class_weight_mode}")

    # Create model
    console.print(f"\n🤖 Creating multimodal model...")
    config = AutoConfig.from_pretrained(args.model_name)
    # `num_labels` is derived from id2label in transformers 5.x; set the real
    # label names and drop the stale literal so the saved config is unambiguous.
    config.id2label = {i: ID2LABEL[i] for i in range(NUM_LABELS)}
    config.label2id = {v: k for k, v in config.id2label.items()}
    config.__dict__.pop("num_labels", None)

    bin_edges = None
    if args.num_encoding == "ple":
        with open(data_dir / "numeric_bins.json") as f:
            bins = json.load(f)
        bin_edges, bin_feature_idx = bins['edges'], bins['feature_idx']
    console.print(f"  Numeric encoding: {args.num_encoding}")

    model = ERPClassifierModel(
        config=config,
        num_encoding=args.num_encoding,
        numeric_bin_edges=bin_edges,
        numeric_bin_feature_idx=bin_feature_idx if bin_edges else None,
        model_name=args.model_name,
        num_depts=num_depts,
        num_continuous=2,  # z_score(net_amount), log1p(net_amount)
        class_weights=class_weights,
        use_class_weights=use_class_weights,
    )

    # transformers 5.x derives `num_labels` from id2label but `to_dict()` reads the
    # legacy __dict__ entry, which is why earlier checkpoints were saved with a
    # literal null. Pin it for a self-describing config.
    model.config.__dict__['num_labels'] = NUM_LABELS

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
        seed=args.seed,
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
        data_collator=MultimodalDataCollator(pad_token_id=tokenizer.pad_token_id),
        compute_metrics=compute_metrics,
        # No EarlyStoppingCallback: with 8 epochs and patience 5 it never fired.
        # load_best_model_at_end already restores the best validation checkpoint.
        optimizers=(optimizer_factory(model), None),  # (optimizer, lr_scheduler)
    )

    # Train
    console.print(f"\n🚀 Starting training...\n")
    train_result = trainer.train()

    # Save
    console.print(f"\n💾 Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    # Keep preprocessing artifacts next to the checkpoint so inference/eval need no hard-coded paths
    for name in ("scaler.json", "dept_vocab.json", "numeric_bins.json"):
        if (data_dir / name).exists():
            shutil.copy(data_dir / name, output_dir / name)

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
            'class_weight_mode': args.class_weight_mode,
            'use_class_weights': use_class_weights,
            'final_metrics': metrics,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }, f, indent=2)

    console.print(f"  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
