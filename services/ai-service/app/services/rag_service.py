"""RAG vector search over knowledge base."""

from __future__ import annotations

from app.services.model_registry import registry


def search_knowledge(query: str, top_k: int = 3) -> list[dict]:
    if registry.rag_embeddings is None or registry.embedding_model is None:
        return []

    query_emb = registry.embedding_model.encode([query], normalize_embeddings=True)[0]
    scores = registry.rag_embeddings @ query_emb
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        doc = registry.rag_documents[int(idx)]
        results.append({
            "score": float(scores[idx]),
            "title": doc.get("title", ""),
            "content": doc.get("content", ""),
            "source_type": doc.get("source_type"),
        })
    return results


def build_grounded_answer(query: str, sources: list[dict]) -> str:
    if not sources:
        return (
            "Thank you for reaching out. A support agent will review your message "
            "and respond shortly."
        )

    best = sources[0]
    return (
        f"Based on our {best.get('source_type', 'knowledge base')} — {best['title']}: "
        f"{best['content']}"
    )
