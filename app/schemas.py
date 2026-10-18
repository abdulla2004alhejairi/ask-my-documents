"""Pydantic request/response models."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Body of POST /ask."""

    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class SourceRef(BaseModel):
    """A cited chunk."""

    source: str
    chunk_index: int
    score: float


class AskResponse(BaseModel):
    """Answer with its supporting sources."""

    answer: str
    sources: list[SourceRef]


class HealthResponse(BaseModel):
    """Body of GET /health."""

    status: str
    index_loaded: bool


class IngestResponse(BaseModel):
    """Body of POST /ingest."""

    documents: int
    chunks: int
