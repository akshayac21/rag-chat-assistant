from typing import Any

import faiss
import numpy as np


class FAISSVectorStore:
    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict[str, Any]] = []

    def build(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata lengths must match")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.metadata = metadata

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[dict[str, Any], float]]:
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized")

        scores, indices = self.index.search(query_embedding, top_k)
        results: list[tuple[dict[str, Any], float]] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index == -1:
                continue
            results.append((self.metadata[int(index)], float(score)))
        return results
