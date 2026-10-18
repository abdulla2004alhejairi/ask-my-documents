"""FastAPI application and routes."""

import logging
import threading
import time
from collections.abc import Awaitable, Callable

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.ingest import ingest_directory
from app.llm import LLMError, generate_answer
from app.retriever import Hit, IndexNotBuiltError, Retriever
from app.schemas import AskRequest, AskResponse, HealthResponse, IngestResponse, SourceRef

logger = logging.getLogger("docsrag")
logging.basicConfig(level=logging.INFO)

Answerer = Callable[[str, list[Hit]], str]

app = FastAPI(title="DocsRAG", description="Ask questions about your documents.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_retriever: Retriever | None = None
_retriever_lock = threading.Lock()


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Log method, path, status and duration of each request."""
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


def get_retriever() -> Retriever:
    """Dependency: the lazily created process-wide retriever."""
    global _retriever
    with _retriever_lock:
        if _retriever is None:
            cfg = get_settings()
            _retriever = Retriever(cfg.index_dir, cfg.embed_model)
        return _retriever


def get_answerer() -> Answerer:
    """Dependency: the function that produces an answer (override in tests)."""
    return generate_answer


@app.get("/health", response_model=HealthResponse)
def health(retriever: Retriever = Depends(get_retriever)) -> HealthResponse:
    """Liveness check; `index_loaded` is true when the index files exist."""
    return HealthResponse(status="ok", index_loaded=retriever.available)


@app.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    retriever: Retriever = Depends(get_retriever),
    answerer: Answerer = Depends(get_answerer),
) -> AskResponse:
    """Retrieve relevant chunks and answer the question from them."""
    try:
        hits = retriever.search(body.question, body.top_k)
    except IndexNotBuiltError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        answer = answerer(body.question, hits)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return AskResponse(
        answer=answer,
        sources=[
            SourceRef(source=h.source, chunk_index=h.chunk_index, score=h.score) for h in hits
        ],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(retriever: Retriever = Depends(get_retriever)) -> IngestResponse:
    """Rebuild the index from DOCS_DIR."""
    try:
        n_docs, n_chunks = ingest_directory()
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    retriever.reset()
    return IngestResponse(documents=n_docs, chunks=n_chunks)
