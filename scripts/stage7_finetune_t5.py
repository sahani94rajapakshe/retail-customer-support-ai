"""
Stage 7: Fine-tune T5-small for response generation
---------------------------------------------------
Output: models/response-t5-small/

Requires: outputs/splits/response_train.csv (from stage3C)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_MAX_LENGTH,
    RESPONSE_MODEL_DIR,
    RESPONSE_MODEL_NAME,
    SPLITS_DIR,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=2000)
    args = parser.parse_args()

    train_path = SPLITS_DIR / "response_train.csv"
    val_path = SPLITS_DIR / "response_val.csv"
    if not train_path.exists():
        raise FileNotFoundError("Run stage3C_extract_response_pairs.py first.")

    train_df = pd.read_csv(train_path).head(args.max_samples)
    val_df = pd.read_csv(val_path).head(max(args.max_samples // 10, 100))

    tokenizer = AutoTokenizer.from_pretrained(RESPONSE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(RESPONSE_MODEL_NAME)

    prefix = "generate response: "

    def preprocess(batch):
        inputs = [prefix + msg for msg in batch["customer_message"]]
        model_inputs = tokenizer(
            inputs, max_length=args.max_length, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            batch["agent_response"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    RESPONSE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(RESPONSE_MODEL_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    print(f"Training T5 response model on {len(train_df)} pairs...")
    trainer.train()
    trainer.save_model(str(RESPONSE_MODEL_DIR))
    tokenizer.save_pretrained(str(RESPONSE_MODEL_DIR))
    print(f"Model saved to {RESPONSE_MODEL_DIR}")


if __name__ == "__main__":
    main()
