"""
Stage 3A: Auto-labeling with Hugging Face Models (weak supervision)
-------------------------------------------------------------------
Outputs:
- outputs/emails_labeled.csv
- outputs/twcs_labeled.csv

Usage:
  python scripts/stage3A_auto_label.py
  python scripts/stage3A_auto_label.py --max-rows 5000
  python scripts/stage3A_auto_label.py --full
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from transformers import pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR,
    DEFAULT_MAX_ROWS,
    INTENT_LABELS,
    OUTPUT_DIR,
    SENTIMENT_MAP,
)


def normalize_sentiment(raw_label: str) -> str:
    return SENTIMENT_MAP.get(raw_label, raw_label)


def classify_batch(classifier, texts, labels=None):
    results = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            results.append(labels[0] if labels else "Neutral")
            continue
        if labels:
            result = classifier(text[:512], labels)
            results.append(result["labels"][0])
        else:
            result = classifier(text[:500])
            results.append(result[0]["label"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Auto-label cleaned datasets")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                        help="Max rows per dataset (default: 5000). Use --full for all.")
    parser.add_argument("--full", action="store_true", help="Process entire datasets")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    emails = pd.read_csv(DATA_DIR / "emails_cleaned.csv")
    tweets = pd.read_csv(DATA_DIR / "twcs_cleaned.csv")

    if not args.full and args.max_rows:
        emails = emails.head(args.max_rows)
        tweets = tweets.head(args.max_rows)

    print(f"Labeling {len(emails)} emails and {len(tweets)} tweets...")
    print(f"Intent classes: {INTENT_LABELS}")

    print("Initializing models (first run downloads weights)...")
    intent_classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
    )
    sentiment_classifier = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
    )

    print("Classifying emails...")
    emails["intent"] = classify_batch(intent_classifier, emails["clean_body"], INTENT_LABELS)
    emails["sentiment"] = [normalize_sentiment(s) for s in
                           classify_batch(sentiment_classifier, emails["clean_body"])]
    emails.to_csv(OUTPUT_DIR / "emails_labeled.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'emails_labeled.csv'}")

    print("Classifying tweets...")
    tweets["intent"] = classify_batch(intent_classifier, tweets["clean_text"], INTENT_LABELS)
    tweets["sentiment"] = [normalize_sentiment(s) for s in
                           classify_batch(sentiment_classifier, tweets["clean_text"])]
    tweets.to_csv(OUTPUT_DIR / "twcs_labeled.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'twcs_labeled.csv'}")

    print("\nIntent distribution (combined):")
    combined = pd.concat([
        emails["intent"],
        tweets["intent"],
    ], ignore_index=True)
    print(combined.value_counts())


if __name__ == "__main__":
    main()
