"""Tests for the retriever against a tiny index."""

import pytest

from app.config import Settings
from app.retriever import IndexNotBuiltError, Retriever


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("how do I cook spaghetti with tomato sauce", "cooking.md"),
        ("which planet is a gas giant", "astronomy.txt"),
        ("who saved the penalty", "football.md"),
    ],
)
def test_top_hit_is_correct_source(
    tiny_settings: Settings, query: str, expected: str
) -> None:
    hits = Retriever(tiny_settings).search(query, 2)
    assert hits[0].source == expected
    assert hits[0].score >= hits[1].score


def test_missing_index_raises(missing_index_dir) -> None:
    retriever = Retriever(Settings(index_dir=missing_index_dir))
    with pytest.raises(IndexNotBuiltError):
        retriever.search("anything", 1)
