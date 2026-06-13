"""Lazy-loaded model registry — fine-tuned models with pretrained fallbacks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from app.config import (
    EMBEDDING_MODEL_NAME,
    INTENT_LABELS,
    INTENT_MODEL_DIR,
    INTENT_MODEL_NAME,
    RAG_INDEX_DIR,
    RESPONSE_MODEL_DIR,
    RESPONSE_MODEL_NAME,
    SENTIMENT_MAP,
    SENTIMENT_MODEL_DIR,
    SENTIMENT_MODEL_NAME,
    ZERO_SHOT_INTENT_MODEL,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelRegistry:
    intent_model: object | None = None
    intent_tokenizer: object | None = None
    intent_mode: str = "not_loaded"
    intent_zero_shot: object | None = None

    sentiment_model: object | None = None
    sentiment_tokenizer: object | None = None
    sentiment_mode: str = "not_loaded"
    sentiment_pipeline: object | None = None

    response_model: object | None = None
    response_tokenizer: object | None = None
    response_mode: str = "not_loaded"

    rag_embeddings: np.ndarray | None = None
    rag_documents: list[dict] = field(default_factory=list)
    rag_mode: str = "not_loaded"
    embedding_model: object | None = None

    def load_all(self) -> None:
        self._load_intent()
        self._load_sentiment()
        self._load_response()
        self._load_rag()
        logger.info(
            "Models loaded — intent=%s sentiment=%s response=%s rag=%s",
            self.intent_mode,
            self.sentiment_mode,
            self.response_mode,
            self.rag_mode,
        )

    def status(self) -> dict[str, str]:
        return {
            "intent": self.intent_mode,
            "sentiment": self.sentiment_mode,
            "response": self.response_mode,
            "rag": self.rag_mode,
        }

    def _load_intent(self) -> None:
        if INTENT_MODEL_DIR.exists() and (INTENT_MODEL_DIR / "config.json").exists():
            self.intent_tokenizer = AutoTokenizer.from_pretrained(str(INTENT_MODEL_DIR))
            self.intent_model = AutoModelForSequenceClassification.from_pretrained(
                str(INTENT_MODEL_DIR)
            )
            self.intent_model.eval()
            self.intent_mode = f"fine-tuned ({INTENT_MODEL_DIR.name})"
            return

        logger.warning("Fine-tuned intent model not found — using zero-shot fallback")
        self.intent_zero_shot = pipeline(
            "zero-shot-classification",
            model=ZERO_SHOT_INTENT_MODEL,
        )
        self.intent_mode = f"zero-shot ({ZERO_SHOT_INTENT_MODEL})"

    def _load_sentiment(self) -> None:
        if SENTIMENT_MODEL_DIR.exists() and (SENTIMENT_MODEL_DIR / "config.json").exists():
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(str(SENTIMENT_MODEL_DIR))
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                str(SENTIMENT_MODEL_DIR)
            )
            self.sentiment_model.eval()
            self.sentiment_mode = f"fine-tuned ({SENTIMENT_MODEL_DIR.name})"
            return

        logger.warning("Fine-tuned sentiment model not found — using pretrained fallback")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL_NAME,
        )
        self.sentiment_mode = f"pretrained ({SENTIMENT_MODEL_NAME})"

    def _load_response(self) -> None:
        if RESPONSE_MODEL_DIR.exists() and (RESPONSE_MODEL_DIR / "config.json").exists():
            self.response_tokenizer = AutoTokenizer.from_pretrained(str(RESPONSE_MODEL_DIR))
            self.response_model = AutoModelForSeq2SeqLM.from_pretrained(str(RESPONSE_MODEL_DIR))
            self.response_model.eval()
            self.response_mode = f"fine-tuned ({RESPONSE_MODEL_DIR.name})"
            return

        self.response_mode = "template (fine-tuned T5 not found)"

    def _load_rag(self) -> None:
        index_file = RAG_INDEX_DIR / "embeddings.npy"
        docs_file = RAG_INDEX_DIR / "documents.json"
        if not index_file.exists() or not docs_file.exists():
            self.rag_mode = "not_available (run stage8_build_rag_index.py)"
            return

        self.rag_embeddings = np.load(index_file)
        self.rag_documents = json.loads(docs_file.read_text(encoding="utf-8"))

        from sentence_transformers import SentenceTransformer

        config_path = RAG_INDEX_DIR / "config.json"
        model_name = EMBEDDING_MODEL_NAME
        if config_path.exists():
            model_name = json.loads(config_path.read_text(encoding="utf-8")).get(
                "embedding_model", EMBEDDING_MODEL_NAME
            )

        self.embedding_model = SentenceTransformer(model_name)
        self.rag_mode = f"indexed ({len(self.rag_documents)} documents)"


registry = ModelRegistry()
