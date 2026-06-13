"""Shared paths, labels, and defaults for the ML pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
OUTPUT_DIR = ROOT / "outputs"
SPLITS_DIR = OUTPUT_DIR / "splits"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

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

# Cardiff NLP Twitter RoBERTa label mapping
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

INTENT_MODEL_DIR = MODELS_DIR / "intent-distilbert"
SENTIMENT_MODEL_DIR = MODELS_DIR / "sentiment-roberta"
RESPONSE_MODEL_DIR = MODELS_DIR / "response-t5-small"
RAG_INDEX_DIR = MODELS_DIR / "rag-index"

DEFAULT_MAX_ROWS = 5000
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 256
