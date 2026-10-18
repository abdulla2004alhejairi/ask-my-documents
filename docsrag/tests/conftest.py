"""Shared fixtures: tiny corpus, isolated settings, and a fake LLM client."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.ingest import build_index, load_documents

FIXTURE_DOCS = {
    "cooking.md": "Pasta is boiled in salted water. Simmer the tomato sauce with garlic "
    "and basil, then toss with spaghetti and parmesan cheese for dinner.",
    "astronomy.txt": "The planet Jupiter is a gas giant with many moons. Telescopes "
    "observe stars, galaxies and nebulae far beyond our solar system.",
    "football.md": "The striker scored a goal in the final minute of the match. The "
    "referee awarded a penalty and the goalkeeper saved the shot.",
}


@pytest.fixture(scope="session")
def tiny_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Settings pointing at a tmp docs dir, with the index already built."""
    root = tmp_path_factory.mktemp("corpus")
    docs = root / "docs"
    docs.mkdir()
    for name, text in FIXTURE_DOCS.items():
        (docs / name).write_text(text, encoding="utf-8")
    settings = Settings(docs_dir=docs, index_dir=root / "index", top_k=2)
    build_index(load_documents(docs), settings)
    return settings


class FakeClient:
    """Stands in for the Anthropic client; optionally raises on call."""

    def __init__(self, answer: str = "mock answer", error: Exception | None = None):
        self._answer = answer
        self._error = error
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id="fake-chat")])
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._generate))

    def _generate(self, **_: object) -> SimpleNamespace:
        if self._error:
            raise self._error
        message = SimpleNamespace(content=self._answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def missing_index_dir(tmp_path: Path) -> Path:
    """An index directory that contains nothing."""
    return tmp_path / "empty"
