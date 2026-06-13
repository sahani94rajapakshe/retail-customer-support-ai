"""
Stage 3B: Prepare unified dataset and train/val/test splits
-------------------------------------------------------------
Outputs:
- outputs/unified_labeled.csv
- outputs/splits/train.csv
- outputs/splits/val.csv
- outputs/splits/test.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INTENT_LABELS, OUTPUT_DIR, SENTIMENT_LABELS, SENTIMENT_MAP, SPLITS_DIR

# Map labels from earlier 3-class runs to the 8-class schema
LEGACY_INTENT_MAP = {
    "WISMO": "OrderTracking",
    "Complaint": "Complaint",
    "Other": "Other",
}


def normalize_sentiment(value: str) -> str:
    return SENTIMENT_MAP.get(value, value)


def normalize_intent(value: str) -> str:
    return LEGACY_INTENT_MAP.get(value, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    emails_path = OUTPUT_DIR / "emails_labeled.csv"
    tweets_path = OUTPUT_DIR / "twcs_labeled.csv"

    if not emails_path.exists() or not tweets_path.exists():
        raise FileNotFoundError(
            "Run stage3A_auto_label.py first to create labeled CSV files."
        )

    emails = pd.read_csv(emails_path)
    tweets = pd.read_csv(tweets_path)

    emails = emails.rename(columns={"clean_body": "text"})
    tweets = tweets.rename(columns={"clean_text": "text"})

    emails["channel"] = "email"
    tweets["channel"] = "chat"

    unified = pd.concat([
        emails[["text", "intent", "sentiment", "channel"]],
        tweets[["text", "intent", "sentiment", "channel"]],
    ], ignore_index=True)

    unified = unified.dropna(subset=["text"])
    unified = unified[unified["text"].str.strip() != ""]
    unified["intent"] = unified["intent"].apply(normalize_intent)
    unified["sentiment"] = unified["sentiment"].apply(normalize_sentiment)
    unified = unified[unified["intent"].isin(INTENT_LABELS)]
    unified = unified[unified["sentiment"].isin(SENTIMENT_LABELS)]

    OUTPUT_DIR.mkdir(exist_ok=True)
    SPLITS_DIR.mkdir(exist_ok=True)
    unified.to_csv(OUTPUT_DIR / "unified_labeled.csv", index=False)

    train_val, test = train_test_split(
        unified,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=unified["intent"] if unified["intent"].value_counts().min() >= 2 else None,
    )
    val_ratio = args.val_size / (1 - args.test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=args.seed,
        stratify=train_val["intent"] if train_val["intent"].value_counts().min() >= 2 else None,
    )

    train.to_csv(SPLITS_DIR / "train.csv", index=False)
    val.to_csv(SPLITS_DIR / "val.csv", index=False)
    test.to_csv(SPLITS_DIR / "test.csv", index=False)

    print(f"Unified dataset: {len(unified)} rows -> {OUTPUT_DIR / 'unified_labeled.csv'}")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print("\nIntent distribution:")
    print(unified["intent"].value_counts())
    print("\nSentiment distribution:")
    print(unified["sentiment"].value_counts())


if __name__ == "__main__":
    main()
