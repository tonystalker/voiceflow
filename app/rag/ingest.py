"""
app/rag/ingest.py

Load faq_knowledge_base.json → embed → upsert into Qdrant.
Run once (or whenever the knowledge base changes):
    python -m app.rag.ingest
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.config import settings

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimension
_FAQ_PATH = Path(__file__).parents[2] / "data" / "faq_knowledge_base.json"


def ingest() -> None:
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    encoder = SentenceTransformer(_EMBED_MODEL)

    # ── Create / recreate collection ─────────────────────────────────────
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in collections:
        client.delete_collection(settings.qdrant_collection)
        logger.info(f"Dropped existing collection '{settings.qdrant_collection}'")

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
    )
    logger.info(f"Created collection '{settings.qdrant_collection}'")

    # ── Load FAQ data ─────────────────────────────────────────────────────
    with open(_FAQ_PATH, encoding="utf-8") as f:
        faqs = json.load(f)

    # ── Embed and upsert ─────────────────────────────────────────────────
    points: list[PointStruct] = []
    for i, item in enumerate(faqs):
        # Combine Q+A for richer embedding signal
        combined = f"Q: {item['question']}\nA: {item['answer']}"
        vector = encoder.encode(combined).tolist()
        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={
                    "text": item["answer"],
                    "question": item["question"],
                    "faq_id": item["id"],
                },
            )
        )

    client.upsert(collection_name=settings.qdrant_collection, points=points)
    logger.info(f"Ingested {len(points)} FAQ items into Qdrant ✓")


if __name__ == "__main__":
    ingest()
