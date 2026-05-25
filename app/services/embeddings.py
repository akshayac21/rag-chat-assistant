import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings


logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.settings.embedding_model)
            self._model = SentenceTransformer(self.settings.embedding_model)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])


embedding_service = EmbeddingService()
