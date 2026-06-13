"""
Stage 4: Fine-tune DistilBERT for intent classification (8 classes)
-------------------------------------------------------------------
Output: models/intent-distilbert/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_MAX_LENGTH,
    INTENT_LABELS,
    INTENT_MODEL_DIR,
    INTENT_MODEL_NAME,
    SPLITS_DIR,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit training rows for quick runs")
    args = parser.parse_args()

    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "val.csv"
    if not train_path.exists():
        raise FileNotFoundError("Run stage3B_prepare_dataset.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    if args.max_samples:
        train_df = train_df.head(args.max_samples)
        val_df = val_df.head(max(args.max_samples // 10, 50))

    label2id = {label: i for i, label in enumerate(INTENT_LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        INTENT_MODEL_NAME,
        num_labels=len(INTENT_LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    train_ds = train_ds.map(lambda x: {"labels": label2id[x["intent"]]})
    val_ds = val_ds.map(lambda x: {"labels": label2id[x["intent"]]})
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)

    INTENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(INTENT_MODEL_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print(f"Training intent model on {len(train_df)} samples...")
    trainer.train()
    trainer.save_model(str(INTENT_MODEL_DIR))
    tokenizer.save_pretrained(str(INTENT_MODEL_DIR))
    print(f"Model saved to {INTENT_MODEL_DIR}")

    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)


if __name__ == "__main__":
    main()
