import logging
from typing import Any

from app.config import get_settings
from app.prompts.prompt_template import build_prompt
from app.services.embeddings import embedding_service
from app.services.llm import llm_service
from app.services.memory import session_memory
from app.services.retriever import Retriever, RetrievedChunk
from app.utils.chunking import chunk_documents
from app.utils.loaders import load_documents
from app.vectorstore.faiss_store import FAISSVectorStore


logger = logging.getLogger(__name__)

FALLBACK_RESPONSE = "I could not find enough information in the knowledge base."


class RAGPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = FAISSVectorStore()
        self.retriever = Retriever(self.store)
        self.document_count = 0
        self.chunk_count = 0
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        documents = load_documents(self.settings.docs_path)
        chunks = chunk_documents(documents)
        texts = [chunk["embedding_text"] for chunk in chunks]
        embeddings = embedding_service.embed_texts(texts)

        self.store.build(embeddings=embeddings, metadata=chunks)
        self.document_count = len(documents)
        self.chunk_count = len(chunks)
        self._initialized = True

        logger.info(
            "Built FAISS index documents=%s chunks=%s",
            self.document_count,
            self.chunk_count,
        )

    async def answer(self, question: str, session_id: str) -> dict[str, Any]:
        if not self._initialized:
            self.initialize()

        retrieved_chunks = self.retriever.retrieve(question)
        best_similarity = retrieved_chunks[0].similarity if retrieved_chunks else 0.0
        logger.info("Best similarity score=%.4f threshold=%.2f", best_similarity, self.settings.similarity_threshold)

        if best_similarity < self.settings.similarity_threshold:
            session_memory.add_exchange(session_id, question, FALLBACK_RESPONSE)
            return self._response(
                answer=FALLBACK_RESPONSE,
                session_id=session_id,
                chunks=retrieved_chunks,
                used_fallback=True,
                similarity_score=best_similarity,
            )

        context = self._format_context(retrieved_chunks)
        history = session_memory.format_history(session_id)
        prompt = build_prompt(
            retrieved_context=context,
            history=history,
            user_question=question,
        )
        answer = await llm_service.generate(prompt)
        if not answer:
            answer = FALLBACK_RESPONSE

        session_memory.add_exchange(session_id, question, answer)
        return self._response(
            answer=answer,
            session_id=session_id,
            chunks=retrieved_chunks,
            used_fallback=False,
            similarity_score=best_similarity,
        )

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(
            (
                f"Source: {chunk.title} ({chunk.source}), chunk {chunk.chunk_id}\n"
                f"Content: {chunk.text}"
            )
            for chunk in chunks
        )

    @staticmethod
    def _response(
        answer: str,
        session_id: str,
        chunks: list[RetrievedChunk],
        used_fallback: bool,
        similarity_score: float,
    ) -> dict[str, Any]:
        return {
            "answer": answer,
            "session_id": session_id,
            "sources": [
                {
                    "title": chunk.title,
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "similarity": round(chunk.similarity, 4),
                }
                for chunk in chunks
            ],
            "used_fallback": used_fallback,
            "similarity_score": round(similarity_score, 4),
        }


rag_pipeline = RAGPipeline()
