"""AI service configuration — paths align with scripts/config.py at repo root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

MODELS_DIR = ROOT / "models"
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"

INTENT_LABELS = [
    "OrderTracking",
    "Refund",
    "ReturnRequest",
    "Complaint",
    "ProductInquiry",
    "PaymentIssue",
    "TechnicalIssue",
    "Other",
]

SENTIMENT_LABELS = ["Positive", "Neutral", "Negative"]

SENTIMENT_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
}

INTENT_MODEL_NAME = "distilbert-base-uncased"
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
RESPONSE_MODEL_NAME = "t5-small"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ZERO_SHOT_INTENT_MODEL = "facebook/bart-large-mnli"

INTENT_MODEL_DIR = MODELS_DIR / "intent-distilbert"
SENTIMENT_MODEL_DIR = MODELS_DIR / "sentiment-roberta"
RESPONSE_MODEL_DIR = MODELS_DIR / "response-t5-small"
RAG_INDEX_DIR = MODELS_DIR / "rag-index"

ESCALATION_SENTIMENT_THRESHOLD = 0.85
MAX_TEXT_LENGTH = 512
