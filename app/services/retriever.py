import logging
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.services.embeddings import embedding_service
from app.vectorstore.faiss_store import FAISSVectorStore


logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    title: str
    source: str
    chunk_id: int
    similarity: float


class Retriever:
    def __init__(self, store: FAISSVectorStore) -> None:
        self.settings = get_settings()
        self.store = store

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        query_embedding = embedding_service.embed_query(query)
        results = self.store.search(query_embedding, top_k=self.settings.top_k * 8)
        chunks: list[RetrievedChunk] = []
        seen: set[tuple[str, str, int]] = set()

        for metadata, similarity in results:
            key = (metadata["title"], metadata["source"], int(metadata["chunk_id"]))
            if key in seen:
                continue
            seen.add(key)
            logger.info(
                "Retrieved chunk title=%s chunk_id=%s vector_type=%s similarity=%.4f",
                metadata["title"],
                metadata["chunk_id"],
                metadata.get("vector_type", "content"),
                similarity,
            )
            chunks.append(self._to_chunk(metadata, similarity))
            if len(chunks) == self.settings.top_k:
                break

        return chunks

    @staticmethod
    def _to_chunk(metadata: dict[str, Any], similarity: float) -> RetrievedChunk:
        return RetrievedChunk(
            text=metadata["text"],
            title=metadata["title"],
            source=metadata["source"],
            chunk_id=int(metadata["chunk_id"]),
            similarity=float(similarity),
        )
