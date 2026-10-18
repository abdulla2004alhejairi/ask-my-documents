"""Prompt construction and LLM call."""

import anthropic

from app.config import get_settings
from app.retriever import Hit

SYSTEM_PROMPT = (
    "Answer only using the provided context. If the answer is not in the context, "
    "say you don't know. Cite sources in square brackets using the file name."
)


class LLMError(RuntimeError):
    """Raised when the LLM call fails."""


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    """Format numbered context blocks `[source#chunk_index] text` plus the question."""
    blocks = [f"[{h.source}#{h.chunk_index}] {h.text}" for h in hits]
    context = "\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate_answer(
    question: str, hits: list[Hit], client: anthropic.Anthropic | None = None
) -> str:
    """Ask the LLM to answer from the hits; wraps SDK errors in LLMError."""
    cfg = get_settings()
    try:
        if client is None:
            key = cfg.anthropic_api_key.get_secret_value() if cfg.anthropic_api_key else None
            client = anthropic.Anthropic(api_key=key)
        message = client.messages.create(
            model=cfg.llm_model,
            max_tokens=cfg.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(question, hits)}],
        )
    except anthropic.AnthropicError as exc:
        raise LLMError(f"LLM request failed: {type(exc).__name__}") from exc
    return "".join(b.text for b in message.content if b.type == "text")
