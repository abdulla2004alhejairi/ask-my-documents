"""API tests with the retriever and LLM replaced by fakes."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.llm import LLMError
from app.main import app, get_answerer, get_retriever
from app.retriever import Hit, Retriever


class FakeRetriever(Retriever):
    """Returns canned hits without touching disk or the embedding model."""

    def __init__(self) -> None:
        super().__init__("unused", "unused")

    @property
    def available(self) -> bool:
        return True

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        return [Hit("cats.md", 0, "Cats purr.", 0.9), Hit("bread.md", 2, "Bread.", 0.5)]


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_retriever] = FakeRetriever
    app.dependency_overrides[get_answerer] = lambda: lambda q, hits: "Mocked [cats.md]"
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "index_loaded": True}


def test_empty_question_is_422(client: TestClient) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_invalid_top_k_is_422(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "hi", "top_k": 11}).status_code == 422


def test_ask_returns_mocked_answer_and_sources(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "Do cats purr?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Mocked [cats.md]"
    assert body["sources"] == [
        {"source": "cats.md", "chunk_index": 0, "score": 0.9},
        {"source": "bread.md", "chunk_index": 2, "score": 0.5},
    ]


def test_ask_503_when_index_missing(tmp_path: Path) -> None:
    app.dependency_overrides[get_retriever] = lambda: Retriever(tmp_path / "none", "m")
    try:
        resp = TestClient(app).post("/ask", json={"question": "hello"})
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 503
    assert "detail" in resp.json()


def test_ask_502_when_llm_fails(client: TestClient) -> None:
    def failing(question: str, hits: list[Hit]) -> str:
        raise LLMError("boom")

    app.dependency_overrides[get_answerer] = lambda: failing
    resp = client.post("/ask", json={"question": "hello"})
    assert resp.status_code == 502
    assert resp.json() == {"detail": "boom"}
