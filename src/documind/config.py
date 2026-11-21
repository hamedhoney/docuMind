from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    app_env: Literal["local", "cloud"] = "local"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"

    embedding_provider: Literal["local", "gemini"] = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    reranker_provider: Literal["local", "none"] = "local"

    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documind"

    frontend_origin: str = "http://localhost:3000"
    public_uploads_enabled: bool = False

    @property
    def qdrant_path(self) -> Path:
        return PROJECT_ROOT / "data" / "qdrant"


@lru_cache
def get_settings() -> Settings:
    return Settings()
