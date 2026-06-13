"""
Stage 3C: Extract customer -> agent response pairs from TWCS for T5 training
-----------------------------------------------------------------------------
Requires raw data/twcs.csv with inbound and text columns.

Outputs:
- outputs/response_pairs.csv
- outputs/splits/response_train.csv
- outputs/splits/response_val.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, OUTPUT_DIR, SPLITS_DIR


def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    twcs_path = DATA_DIR / "twcs.csv"
    if not twcs_path.exists():
        raise FileNotFoundError(
            f"{twcs_path} not found. Download TWCS dataset and place in data/."
        )

    df = pd.read_csv(twcs_path)
    required = {"text", "inbound", "in_response_to_tweet_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"twcs.csv must contain columns: {required}")

    df["tweet_id"] = df.get("tweet_id", df.index)
    inbound = df[df["inbound"] == True].copy()  # noqa: E712
    outbound = df[df["inbound"] == False].copy()  # noqa: E712

    outbound = outbound.set_index("in_response_to_tweet_id", drop=False)
    pairs = []

    for _, row in inbound.iterrows():
        response_id = row.get("in_response_to_tweet_id")
        if pd.isna(response_id) or response_id not in outbound.index:
            continue
        customer_text = clean_tweet(row["text"])
        agent_text = clean_tweet(outbound.loc[response_id, "text"])
        if len(customer_text) < 10 or len(agent_text) < 10:
            continue
        pairs.append({
            "customer_message": customer_text,
            "agent_response": agent_text,
            "intent": "Other",
        })

    pairs_df = pd.DataFrame(pairs).drop_duplicates(subset=["customer_message", "agent_response"])
    if args.max_rows:
        pairs_df = pairs_df.head(args.max_rows)

    OUTPUT_DIR.mkdir(exist_ok=True)
    SPLITS_DIR.mkdir(exist_ok=True)
    pairs_df.to_csv(OUTPUT_DIR / "response_pairs.csv", index=False)

    train, val = train_test_split(pairs_df, test_size=0.1, random_state=args.seed)
    train.to_csv(SPLITS_DIR / "response_train.csv", index=False)
    val.to_csv(SPLITS_DIR / "response_val.csv", index=False)

    print(f"Extracted {len(pairs_df)} response pairs -> {OUTPUT_DIR / 'response_pairs.csv'}")
    print(f"  Train: {len(train)} | Val: {len(val)}")


if __name__ == "__main__":
    main()
