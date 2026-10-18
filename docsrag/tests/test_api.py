"""API tests using dependency overrides; no real LLM or network calls."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import LLMError
from app.main import app, get_llm_client, get_retriever
from app.retriever import Retriever
from tests.conftest import FakeClient


@pytest.fixture
def client(tiny_settings: Settings) -> Iterator[TestClient]:
    app.dependency_overrides[get_retriever] = lambda: Retriever(tiny_settings)
    app.dependency_overrides[get_llm_client] = lambda: FakeClient("mock answer")
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert isinstance(resp.json()["index_loaded"], bool)


def test_chat_ui_served(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "DocsRAG" in resp.text


@pytest.mark.parametrize(
    "body",
    [{"question": ""}, {"question": "x" * 1001}, {"question": "hi", "top_k": 11}, {}],
)
def test_invalid_input_422(client: TestClient, body: dict) -> None:
    assert client.post("/ask", json=body).status_code == 422


def test_ask_returns_mock_answer_and_sources(client: TestClient) -> None:
    resp = client.post(
        "/ask", json={"question": "which planet is a gas giant", "top_k": 2}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "mock answer"
    assert len(data["sources"]) == 2
    assert data["sources"][0]["source"] == "astronomy.txt"
    assert set(data["sources"][0]) == {"source", "chunk_index", "score"}


def test_ask_503_when_index_missing(client: TestClient, missing_index_dir) -> None:
    app.dependency_overrides[get_retriever] = lambda: Retriever(
        Settings(index_dir=missing_index_dir)
    )
    resp = client.post("/ask", json={"question": "anything"})
    assert resp.status_code == 503
    assert "detail" in resp.json()


def test_ask_502_when_llm_fails(client: TestClient) -> None:
    failing = FakeClient(error=LLMError("boom"))
    app.dependency_overrides[get_llm_client] = lambda: failing
    resp = client.post("/ask", json={"question": "which planet is a gas giant"})
    assert resp.status_code == 502
    assert "detail" in resp.json()
