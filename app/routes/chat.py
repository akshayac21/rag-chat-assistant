import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.llm import InvalidAPIKeyError, LLMRateLimitError, LLMTimeoutError
from app.services.rag_pipeline import rag_pipeline


logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1, max_length=128)


class Source(BaseModel):
    title: str
    source: str
    chunk_id: int
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[Source]
    used_fallback: bool
    similarity_score: float


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> dict[str, Any]:
    try:
        return await rag_pipeline.answer(
            question=payload.message.strip(),
            session_id=payload.session_id.strip(),
        )
    except InvalidAPIKeyError as exc:
        logger.exception("Invalid OpenRouter API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid OpenRouter API key. Check OPENROUTER_API_KEY.",
        ) from exc
    except LLMRateLimitError as exc:
        logger.exception("OpenRouter rate limit reached")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit reached. Please try again shortly.",
        ) from exc
    except LLMTimeoutError as exc:
        logger.exception("OpenRouter request timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The model request timed out. Please try again.",
        ) from exc
    except RuntimeError as exc:
        logger.exception("Chat pipeline runtime error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
