"""
Stage 8: Build RAG knowledge index with Sentence Transformers
-------------------------------------------------------------
Output: models/rag-index/ (embeddings.npy, documents.json, config.json)

Usage:
  python scripts/stage8_build_rag_index.py
  python scripts/stage8_build_rag_index.py --query "Does the laptop support 32GB RAM?"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EMBEDDING_MODEL_NAME, KNOWLEDGE_DIR, RAG_INDEX_DIR


def load_documents(knowledge_dir: Path) -> list[dict]:
    documents = []
    for path in sorted(knowledge_dir.glob("*.json")):
        items = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(items, list):
            documents.extend(items)
        else:
            documents.append(items)
    return documents


def build_index(documents: list[dict], model_name: str) -> tuple[np.ndarray, list[dict]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [f"{d['title']}. {d['content']}" for d in documents]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return embeddings, documents


def search(query: str, embeddings: np.ndarray, documents: list[dict], model_name: str, top_k: int = 3):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_emb
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"score": float(scores[i]), "title": documents[i]["title"], "content": documents[i]["content"]}
        for i in top_indices
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Test a search query after building")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {KNOWLEDGE_DIR}")

    documents = load_documents(KNOWLEDGE_DIR)
    if not documents:
        raise ValueError(f"No documents found in {KNOWLEDGE_DIR}")

    print(f"Building RAG index from {len(documents)} documents...")
    embeddings, documents = build_index(documents, EMBEDDING_MODEL_NAME)

    RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RAG_INDEX_DIR / "embeddings.npy", embeddings)
    (RAG_INDEX_DIR / "documents.json").write_text(
        json.dumps(documents, indent=2), encoding="utf-8"
    )
    (RAG_INDEX_DIR / "config.json").write_text(
        json.dumps({"embedding_model": EMBEDDING_MODEL_NAME, "num_documents": len(documents)}),
        encoding="utf-8",
    )
    print(f"RAG index saved to {RAG_INDEX_DIR}")

    if args.query:
        results = search(args.query, embeddings, documents, EMBEDDING_MODEL_NAME, args.top_k)
        print(f"\nTop results for: {args.query!r}")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] {r['title']}: {r['content'][:120]}...")


if __name__ == "__main__":
    main()
