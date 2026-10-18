"""Application settings loaded from environment variables."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable via env vars or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    docs_dir: str = "data/docs"
    index_dir: str = "storage"
    embed_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    llm_model: str = "claude-sonnet-5-5"
    max_tokens: int = 600
    anthropic_api_key: SecretStr | None = None


def get_settings() -> Settings:
    """Return settings read fresh from the current environment."""
    return Settings()
