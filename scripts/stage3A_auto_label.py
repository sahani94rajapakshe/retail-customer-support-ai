"""
Stage 3A: Auto-labeling with Hugging Face Models
-----------------------------------------------
Outputs:
- outputs/emails_labeled.csv
- outputs/twcs_labeled.csv
"""

from pathlib import Path

import pandas as pd
from transformers import pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

emails = pd.read_csv(DATA_DIR / "emails_cleaned.csv")
tweets = pd.read_csv(DATA_DIR / "twcs_cleaned.csv")

print("Loaded cleaned datasets.")

# Sample for testing (remove .head(50) for full run)
emails = emails.head(50)
tweets = tweets.head(50)

print("Initializing models...")
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

intent_labels = ["WISMO", "Complaint", "Other"]

def classify_intent(text):
    if not isinstance(text, str) or text.strip() == "":
        return "Other"
    result = intent_classifier(text, intent_labels)
    return result["labels"][0]

def classify_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "Neutral"
    truncated_text = text[:500]
    result = sentiment_classifier(truncated_text)
    return result[0]["label"]

print("Classifying emails...")
emails["intent"] = emails["clean_body"].apply(classify_intent)
emails["sentiment"] = emails["clean_body"].apply(classify_sentiment)
emails.to_csv(OUTPUT_DIR / "emails_labeled.csv", index=False)
print(f"Emails labeled and saved to {OUTPUT_DIR / 'emails_labeled.csv'}")

print("Classifying tweets...")
tweets["intent"] = tweets["clean_text"].apply(classify_intent)
tweets["sentiment"] = tweets["clean_text"].apply(classify_sentiment)
tweets.to_csv(OUTPUT_DIR / "twcs_labeled.csv", index=False)
print(f"Tweets labeled and saved to {OUTPUT_DIR / 'twcs_labeled.csv'}")
