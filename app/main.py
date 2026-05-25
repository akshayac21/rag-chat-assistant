import logging
import sys
from contextlib import asynccontextmanager
from json import dumps
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Allows both `uvicorn app.main:app` and direct `python app/main.py` execution.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.routes.chat import router as chat_router
from app.services.rag_pipeline import rag_pipeline


settings = get_settings()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return dumps(payload)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=settings.log_level.upper(), handlers=[handler], force=True)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    rag_pipeline.initialize()
    logger.info("RAG pipeline initialized")
    yield


app = FastAPI(
    title="RAG Chat Assistant API",
    description="FastAPI RAG chatbot using Sentence Transformers, FAISS, and OpenRouter.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health() -> dict[str, str | int]:
    return {
        "status": "healthy",
        "documents_indexed": rag_pipeline.document_count,
        "chunks_indexed": rag_pipeline.chunk_count,
    }


app.include_router(chat_router, prefix="/api", tags=["chat"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
