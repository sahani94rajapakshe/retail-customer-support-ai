"""Intent classification service."""

from __future__ import annotations

import torch

from app.config import INTENT_LABELS, MAX_TEXT_LENGTH
from app.services.model_registry import registry


def classify_intent(text: str) -> tuple[str, float, str]:
    text = text.strip()[:MAX_TEXT_LENGTH]

    if registry.intent_model is not None:
        tokenizer = registry.intent_tokenizer
        model = registry.intent_model
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TEXT_LENGTH)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            idx = int(probs.argmax())
            confidence = float(probs[idx])
            intent = model.config.id2label[idx]
        return intent, confidence, registry.intent_mode

    result = registry.intent_zero_shot(text, INTENT_LABELS)
    intent = result["labels"][0]
    scores = result["scores"]
    confidence = float(scores[0]) if scores else 0.0
    return intent, confidence, registry.intent_mode
