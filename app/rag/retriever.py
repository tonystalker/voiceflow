"""
app/rag/retriever.py

Wraps Qdrant similarity search for the FAQ knowledge base.
Returns top-k chunks with cosine scores.
Score threshold < 0.5 → signal fallback to caller.
"""

from __future__ import annotations

from typing import List, Dict, Any

from loguru import logger

try:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    HAS_RAG = True
except ImportError:
    logger.warning("RAG dependencies not found. RAG retriever will be disabled.")
    HAS_RAG = False

from app.config import settings

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_FALLBACK_THRESHOLD = 0.30   # lowered: minor transcription errors drop scores


class RAGRetriever:
    def __init__(self) -> None:
        if HAS_RAG:
            self._client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
            self._encoder = SentenceTransformer(_EMBED_MODEL)
            logger.info(f"RAG retriever initialised (collection={settings.qdrant_collection})")
        else:
            self._client = None
            self._encoder = None

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search the FAQ collection.

        Returns list of dicts:
            {"text": str, "score": float, "metadata": dict}

        If best score < threshold, list is returned with `low_confidence=True`
        on the first item so the agent can route to fallback.
        """
        if not HAS_RAG:
            return []

        vector = self._encoder.encode(query).tolist()

        results = self._client.search(
            collection_name=settings.qdrant_collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )

        chunks = []
        for hit in results:
            chunks.append(
                {
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                }
            )

        if chunks and chunks[0]["score"] < _FALLBACK_THRESHOLD:
            logger.info(
                f"[RAG] Low confidence ({chunks[0]['score']:.2f}) for query: '{query}'"
            )
            chunks[0]["low_confidence"] = True
        else:
            for c in chunks:
                c["low_confidence"] = False

        logger.debug(f"[RAG] top hit score={chunks[0]['score']:.2f}" if chunks else "[RAG] no results")
        return chunks
