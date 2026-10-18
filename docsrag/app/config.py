"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration; every field can be overridden by an env var."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    docs_dir: Path = Path("data/docs")
    index_dir: Path = Path("storage")
    embed_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    llm_model: str | None = None
    llm_base_url: str | None = None
    max_tokens: int = 600
    llm_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
