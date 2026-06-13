"""Sentiment analysis service."""

from __future__ import annotations

import torch

from app.config import MAX_TEXT_LENGTH, SENTIMENT_MAP
from app.services.model_registry import registry


def _normalize(label: str) -> str:
    return SENTIMENT_MAP.get(label, label)


def analyze_sentiment(text: str) -> tuple[str, float, str]:
    text = text.strip()[:MAX_TEXT_LENGTH]

    if registry.sentiment_model is not None:
        tokenizer = registry.sentiment_tokenizer
        model = registry.sentiment_model
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TEXT_LENGTH)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            idx = int(probs.argmax())
            score = float(probs[idx])
            sentiment = _normalize(model.config.id2label[idx])
        return sentiment, score, registry.sentiment_mode

    result = registry.sentiment_pipeline(text[:500])[0]
    sentiment = _normalize(result["label"])
    score = float(result["score"])
    return sentiment, score, registry.sentiment_mode
