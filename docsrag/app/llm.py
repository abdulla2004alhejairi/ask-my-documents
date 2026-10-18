"""Prompt construction and LLM call through any OpenAI-compatible chat API."""

from functools import lru_cache
from typing import Protocol

import openai

from app.config import get_settings
from app.retriever import Hit

SYSTEM_PROMPT = (
    "Answer only using the provided context. If the answer is not in the context, "
    "say you don't know. Cite sources in square brackets using the file name."
)
# Specific prefixes must come before the generic "sk-".
KEY_PREFIX_ENDPOINTS = (
    ("AIza", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    ("AQ.", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    ("xai-", "https://api.x.ai/v1"),
    ("sk-ant-", "https://api.anthropic.com/v1/"),
    ("sk-or-", "https://openrouter.ai/api/v1"),
    ("gsk_", "https://api.groq.com/openai/v1"),
    ("sk-", "https://api.openai.com/v1"),
)
# Model ids containing these words are not text chat models.
NON_CHAT_WORDS = (
    "embed", "image", "imagen", "vision", "audio", "tts", "whisper", "dall",
    "moderation", "veo", "live", "transcribe", "realtime", "aqa", "guard",
)  # fmt: skip
# Cheap, fast general models are preferred when several are available.
PREFERRED_WORDS = ("flash", "mini", "haiku", "instant", "small", "grok")


class LLMError(RuntimeError):
    """Raised when the LLM call fails."""


class LLMClient(Protocol):
    """Minimal interface of the OpenAI-compatible client used here."""

    chat: openai.resources.chat.Chat
    models: openai.resources.Models


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    """Format numbered context blocks followed by the question."""
    blocks = [
        f"{i}. [{h.source}#{h.chunk_index}] {h.text}" for i, h in enumerate(hits, 1)
    ]
    return "Context:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def detect_base_url(api_key: str) -> str:
    """Infer the provider endpoint from the API key prefix."""
    for prefix, url in KEY_PREFIX_ENDPOINTS:
        if api_key.startswith(prefix):
            return url
    raise LLMError("Unknown API key format; set LLM_BASE_URL for your provider")


def get_client() -> openai.OpenAI:
    """Create a client for the configured or key-detected endpoint."""
    settings = get_settings()
    if settings.llm_api_key is None:
        raise LLMError("LLM_API_KEY is not configured")
    key = settings.llm_api_key.get_secret_value()
    return openai.OpenAI(
        api_key=key, base_url=settings.llm_base_url or detect_base_url(key)
    )


def pick_model(model_ids: list[str]) -> str:
    """Choose a general-purpose chat model from a provider's model list."""
    ids = [m.removeprefix("models/") for m in model_ids]
    chat_ids = [m for m in ids if not any(w in m.lower() for w in NON_CHAT_WORDS)]
    # "-latest" aliases keep working when pinned versions are retired.
    chat_ids.sort(key=lambda m: not m.endswith("-latest"))
    for preferred in PREFERRED_WORDS:
        for model in chat_ids:
            if preferred in model.lower():
                return model
    if chat_ids:
        return chat_ids[0]
    raise LLMError("No usable chat model found; set LLM_MODEL explicitly")


@lru_cache
def detect_model(client: LLMClient) -> str:
    """Ask the provider which models this API key can use and pick one."""
    return pick_model([m.id for m in client.models.list().data])


def generate_answer(
    question: str, hits: list[Hit], client: LLMClient | None = None
) -> str:
    """Ask the LLM to answer from the hits only; wrap SDK errors in LLMError."""
    settings = get_settings()
    try:
        client = client or get_client()
        response = client.chat.completions.create(
            model=settings.llm_model or detect_model(client),
            max_tokens=settings.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, hits)},
            ],
        )
    except openai.OpenAIError as exc:
        raise LLMError(f"LLM request failed: {type(exc).__name__}") from exc
    return response.choices[0].message.content or ""
