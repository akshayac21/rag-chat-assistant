from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
        default="google/gemini-2.0-flash-exp:free",
        alias="OPENROUTER_MODEL",
    )
    openrouter_site_url: str = Field(default="http://localhost:8000", alias="OPENROUTER_SITE_URL")
    openrouter_app_name: str = Field(default="RAG Chat Assistant", alias="OPENROUTER_APP_NAME")
    openrouter_api_url: str = Field(
        default="https://openrouter.ai/api/v1/chat/completions",
        alias="OPENROUTER_API_URL",
    )
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    similarity_threshold: float = Field(default=0.65, alias="SIMILARITY_THRESHOLD")
    top_k: int = Field(default=3, alias="TOP_K")
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173,http://127.0.0.1:5500",
        alias="ALLOWED_ORIGINS",
    )
    request_timeout_seconds: int = Field(default=30, alias="REQUEST_TIMEOUT_SECONDS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    docs_path: Path = BASE_DIR / "data" / "docs.json"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
