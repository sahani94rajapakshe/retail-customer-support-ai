"""FastAPI AI microservice for retail customer support."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.schemas import (
    AnalyzeResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    IntentResult,
    RagAnswerRequest,
    RagAnswerResponse,
    RagDocument,
    RagSearchRequest,
    RagSearchResponse,
    SentimentResult,
    TextRequest,
)
from app.services.escalation_service import check_escalation
from app.services.intent_service import classify_intent
from app.services.model_registry import registry
from app.services.rag_service import build_grounded_answer, search_knowledge
from app.services.response_service import generate_response
from app.services.sentiment_service import analyze_sentiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading AI models...")
    registry.load_all()
    yield
    logger.info("Shutting down AI service")


app = FastAPI(
    title="Retail Support AI Service",
    description=(
        "Intent classification, sentiment analysis, RAG search, "
        "response generation, and escalation logic for omnichannel customer support."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(status="ok", models=registry.status())


@app.post("/classify/intent", response_model=IntentResult, tags=["Classification"])
def classify_intent_endpoint(body: TextRequest):
    intent, confidence, model = classify_intent(body.text)
    return IntentResult(intent=intent, confidence=confidence, model=model)


@app.post("/classify/sentiment", response_model=SentimentResult, tags=["Classification"])
def classify_sentiment_endpoint(body: TextRequest):
    sentiment, score, model = analyze_sentiment(body.text)
    return SentimentResult(sentiment=sentiment, score=score, model=model)


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Classification"])
def analyze_message(body: TextRequest):
    intent, intent_confidence, intent_model = classify_intent(body.text)
    sentiment, sentiment_score, sentiment_model = analyze_sentiment(body.text)
    escalate, reason = check_escalation(intent, sentiment, sentiment_score)

    return AnalyzeResponse(
        text=body.text,
        intent=intent,
        intent_confidence=intent_confidence,
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        escalate=escalate,
        escalation_reason=reason,
        models={"intent": intent_model, "sentiment": sentiment_model},
    )


@app.post("/generate/response", response_model=GenerateResponse, tags=["Generation"])
def generate_response_endpoint(body: GenerateRequest):
    response, intent, model = generate_response(body.text, body.intent, body.context)
    return GenerateResponse(response=response, intent=intent, model=model)


@app.post("/rag/search", response_model=RagSearchResponse, tags=["RAG"])
def rag_search(body: RagSearchRequest):
    if registry.rag_mode.startswith("not_available"):
        raise HTTPException(
            status_code=503,
            detail="RAG index not available. Run scripts/stage8_build_rag_index.py first.",
        )
    results = search_knowledge(body.query, body.top_k)
    return RagSearchResponse(
        query=body.query,
        results=[RagDocument(**r) for r in results],
    )


@app.post("/rag/answer", response_model=RagAnswerResponse, tags=["RAG"])
def rag_answer(body: RagAnswerRequest):
    if registry.rag_mode.startswith("not_available"):
        raise HTTPException(
            status_code=503,
            detail="RAG index not available. Run scripts/stage8_build_rag_index.py first.",
        )
    sources = search_knowledge(body.query, body.top_k)
    answer = build_grounded_answer(body.query, sources)
    return RagAnswerResponse(
        query=body.query,
        answer=answer,
        sources=[RagDocument(**s) for s in sources],
        model=f"rag + {registry.rag_mode}",
    )
