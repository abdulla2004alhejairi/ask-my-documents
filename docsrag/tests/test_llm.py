"""Tests for automatic model selection."""

import pytest

from app.llm import LLMError, detect_base_url, pick_model


@pytest.mark.parametrize(
    ("key", "host"),
    [
        ("AIzaSyExample", "generativelanguage.googleapis.com"),
        ("xai-abc", "api.x.ai"),
        ("sk-ant-abc", "api.anthropic.com"),
        ("sk-or-abc", "openrouter.ai"),
        ("gsk_abc", "api.groq.com"),
        ("sk-proj-abc", "api.openai.com"),
    ],
)
def test_base_url_detected_from_key(key: str, host: str) -> None:
    assert host in detect_base_url(key)


def test_unknown_key_format_raises() -> None:
    with pytest.raises(LLMError):
        detect_base_url("mystery-key")


def test_prefers_fast_chat_model_and_skips_non_chat() -> None:
    ids = [
        "models/text-embedding-004",
        "models/gemini-2.5-pro",
        "models/imagen-3.0",
        "models/gemini-2.5-flash",
    ]
    assert pick_model(ids) == "gemini-2.5-flash"


def test_prefers_latest_alias() -> None:
    ids = ["models/gemini-2.5-flash", "models/gemini-flash-latest"]
    assert pick_model(ids) == "gemini-flash-latest"


def test_falls_back_to_first_chat_model() -> None:
    assert pick_model(["whisper-1", "gpt-4o", "gpt-4.1"]) == "gpt-4o"


def test_no_chat_model_raises() -> None:
    with pytest.raises(LLMError):
        pick_model(["text-embedding-3-small"])
