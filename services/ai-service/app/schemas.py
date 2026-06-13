from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, examples=["Where is my order #12345?"])


class IntentResult(BaseModel):
    intent: str
    confidence: float
    model: str


class SentimentResult(BaseModel):
    sentiment: str
    score: float
    model: str


class AnalyzeResponse(BaseModel):
    text: str
    intent: str
    intent_confidence: float
    sentiment: str
    sentiment_score: float
    escalate: bool
    escalation_reason: str | None
    models: dict[str, str]


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    intent: str | None = None
    context: str | None = None


class GenerateResponse(BaseModel):
    response: str
    intent: str
    model: str


class RagSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class RagDocument(BaseModel):
    score: float
    title: str
    content: str
    source_type: str | None = None


class RagSearchResponse(BaseModel):
    query: str
    results: list[RagDocument]


class RagAnswerRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class RagAnswerResponse(BaseModel):
    query: str
    answer: str
    sources: list[RagDocument]
    model: str


class HealthResponse(BaseModel):
    status: str
    models: dict[str, str]
