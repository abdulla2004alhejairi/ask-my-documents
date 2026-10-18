"""FastAPI application and routes."""

import logging
import time
from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app import ingest as ingest_module
from app.config import get_settings
from app.llm import LLMClient, LLMError, generate_answer
from app.retriever import IndexNotBuiltError, Retriever
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestResponse,
    SourceRef,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docsrag")
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="DocsRAG", description="Ask questions about your documents.")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Log method, path, status and duration of every request."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@lru_cache
def get_retriever() -> Retriever:
    """Return the process-wide lazily loaded retriever."""
    return Retriever(get_settings())


def get_llm_client() -> LLMClient | None:
    """Dependency providing the LLM client; None means build the default lazily."""
    return None


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page chat UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health(retriever: Retriever = Depends(get_retriever)) -> HealthResponse:
    """Report liveness and whether the index is loaded in memory."""
    return HealthResponse(status="ok", index_loaded=retriever.loaded)


@app.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    retriever: Retriever = Depends(get_retriever),
    client: LLMClient | None = Depends(get_llm_client),
) -> AskResponse:
    """Retrieve relevant chunks and answer the question from them."""
    try:
        hits = retriever.search(body.question, body.top_k)
    except IndexNotBuiltError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        answer = generate_answer(body.question, hits, client)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return AskResponse(
        answer=answer,
        sources=[
            SourceRef(source=h.source, chunk_index=h.chunk_index, score=h.score)
            for h in hits
        ],
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest() -> IngestResponse:
    """Rebuild the index from DOCS_DIR and reload the retriever."""
    try:
        documents, chunks = await run_in_threadpool(ingest_module.ingest)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    get_retriever.cache_clear()
    return IngestResponse(documents=documents, chunks=chunks)
