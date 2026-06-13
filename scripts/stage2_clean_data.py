"""
Stage 2: Data Cleaning for Retail Customer Support
-------------------------------------------------
Purpose:
- Clean Enron Email dataset (remove headers, PII, normalize text)
- Prepare Twitter Customer Support dataset (basic cleaning)
Outputs:
- data/emails_cleaned.csv
- data/twcs_cleaned.csv
"""

from pathlib import Path

import pandas as pd
import re

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

print("Loading Enron Email dataset...")
emails = pd.read_csv(DATA_DIR / "emails.csv")

def extract_body(email_text):
    if not isinstance(email_text, str):
        return ""
    parts = email_text.split('\n\n', 1)
    return parts[1].strip() if len(parts) > 1 else email_text.strip()

def clean_email_body(text):
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'\+?\d[\d\s-]{7,}\d', '[PHONE]', text)
    text = re.sub(r'[^a-zA-Z0-9,.!? ]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

emails['body'] = emails['message'].apply(extract_body)
emails['clean_body'] = emails['body'].apply(clean_email_body)
emails[['clean_body']].to_csv(DATA_DIR / "emails_cleaned.csv", index=False)
print(f"Emails cleaned and saved to {DATA_DIR / 'emails_cleaned.csv'}")

print("\nLoading Twitter Customer Support dataset...")
tweets = pd.read_csv(DATA_DIR / "twcs.csv")

def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

tweets['clean_text'] = tweets['text'].apply(clean_tweet)
tweets[['clean_text']].to_csv(DATA_DIR / "twcs_cleaned.csv", index=False)
print(f"Tweets cleaned and saved to {DATA_DIR / 'twcs_cleaned.csv'}")
