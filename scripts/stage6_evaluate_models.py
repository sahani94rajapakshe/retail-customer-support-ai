"""
Stage 6: Evaluate fine-tuned models on test set
-----------------------------------------------
Output: outputs/reports/evaluation_report.json
        outputs/reports/evaluation_report.txt
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    INTENT_LABELS,
    INTENT_MODEL_DIR,
    INTENT_MODEL_NAME,
    OUTPUT_DIR,
    REPORTS_DIR,
    SENTIMENT_LABELS,
    SENTIMENT_MODEL_DIR,
    SENTIMENT_MODEL_NAME,
    SPLITS_DIR,
)


def evaluate_classifier(model_dir, model_fallback, texts, true_labels, label_list, task_name):
    if model_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        id2label = model.config.id2label
        preds = []
        for text in texts:
            inputs = tokenizer(text[:512], return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = model(**inputs).logits
            preds.append(id2label[int(logits.argmax(-1))])
        model_used = str(model_dir)
    else:
        print(f"  {task_name}: fine-tuned model not found, using baseline zero-shot/pretrained")
        if task_name == "intent":
            clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            preds = [clf(t[:512], label_list)["labels"][0] for t in texts]
        else:
            clf = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME)
            from config import SENTIMENT_MAP
            preds = [SENTIMENT_MAP.get(clf(t[:500])[0]["label"], "Neutral") for t in texts]
        model_used = f"baseline ({model_fallback})"

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
    report = classification_report(true_labels, preds, zero_division=0)

    return {
        "task": task_name,
        "model": model_used,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "classification_report": report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    test_path = SPLITS_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError("Run stage3B_prepare_dataset.py first.")

    test_df = pd.read_csv(test_path)
    if args.max_samples:
        test_df = test_df.head(args.max_samples)

    texts = test_df["text"].tolist()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating on {len(test_df)} test samples...")
    intent_results = evaluate_classifier(
        INTENT_MODEL_DIR, INTENT_MODEL_NAME, texts,
        test_df["intent"].tolist(), INTENT_LABELS, "intent",
    )
    sentiment_results = evaluate_classifier(
        SENTIMENT_MODEL_DIR, SENTIMENT_MODEL_NAME, texts,
        test_df["sentiment"].tolist(), SENTIMENT_LABELS, "sentiment",
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_samples": len(test_df),
        "intent": {k: v for k, v in intent_results.items() if k != "classification_report"},
        "sentiment": {k: v for k, v in sentiment_results.items() if k != "classification_report"},
    }

    json_path = REPORTS_DIR / "evaluation_report.json"
    txt_path = REPORTS_DIR / "evaluation_report.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    txt_lines = [
        "Evaluation Report",
        "=" * 50,
        f"Test samples: {len(test_df)}",
        "",
        "INTENT",
        f"  Model:    {intent_results['model']}",
        f"  Accuracy: {intent_results['accuracy']}",
        f"  F1 macro: {intent_results['f1_macro']}",
        intent_results["classification_report"],
        "",
        "SENTIMENT",
        f"  Model:    {sentiment_results['model']}",
        f"  Accuracy: {sentiment_results['accuracy']}",
        f"  F1 macro: {sentiment_results['f1_macro']}",
        sentiment_results["classification_report"],
    ]
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    print(f"\nReports saved to {REPORTS_DIR}")
    print(f"Intent  — accuracy: {intent_results['accuracy']}, F1: {intent_results['f1_macro']}")
    print(f"Sentiment — accuracy: {sentiment_results['accuracy']}, F1: {sentiment_results['f1_macro']}")


if __name__ == "__main__":
    main()
