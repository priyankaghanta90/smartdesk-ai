"""LLM + embedding factories with retry and model fallback.

Every chat call goes: primary model (max_retries with exponential backoff inside the
OpenAI client) -> fallback model. Callers catch `LLMUnavailable` for a graceful reply.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from functools import lru_cache

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import get_settings


class LLMUnavailable(RuntimeError):
    """Both the primary and fallback model failed."""


def _chat_model(model: str, temperature: float) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        timeout=s.llm_timeout_s,
        max_retries=s.llm_max_retries,
        api_key=s.openai_api_key or None,
    )


def chat(temperature: float = 0.0) -> Runnable:
    s = get_settings()
    primary = _chat_model(s.openai_chat_model, temperature)
    fallback = _chat_model(s.openai_fallback_model, temperature)
    return primary.with_fallbacks([fallback])


def structured(schema, temperature: float = 0.0) -> Runnable:
    """Structured-output runnable (pydantic schema) with the same fallback chain."""
    s = get_settings()
    primary = _chat_model(s.openai_chat_model, temperature).with_structured_output(schema)
    fallback = _chat_model(s.openai_fallback_model, temperature).with_structured_output(schema)
    return primary.with_fallbacks([fallback])


def invoke_safely(runnable: Runnable, payload):
    try:
        return runnable.invoke(payload)
    except Exception as exc:  # network errors, auth errors, rate limits after retries
        raise LLMUnavailable(str(exc)) from exc


_QUERY_MEMO: OrderedDict[tuple[str, str], tuple[float, ...]] = OrderedDict()
_QUERY_MEMO_MAX = 512
_QUERY_MEMO_LOCK = threading.Lock()


class _MemoEmbeddings(OpenAIEmbeddings):
    """Remembers recent query embeddings: on a cache miss the semantic cache and Chroma embed the same
    question back to back, which would otherwise be two identical OpenAI calls."""

    def embed_query(self, text: str, **kwargs) -> list[float]:
        key = (self.model, text)
        with _QUERY_MEMO_LOCK:
            if (hit := _QUERY_MEMO.get(key)) is not None:
                _QUERY_MEMO.move_to_end(key)
                return list(hit)
        vec = super().embed_query(text, **kwargs)
        with _QUERY_MEMO_LOCK:
            _QUERY_MEMO[key] = tuple(vec)
            if len(_QUERY_MEMO) > _QUERY_MEMO_MAX:
                _QUERY_MEMO.popitem(last=False)
        return vec


@lru_cache
def embeddings() -> OpenAIEmbeddings:
    s = get_settings()
    return _MemoEmbeddings(
        model=s.openai_embedding_model,
        api_key=s.openai_api_key or None,
        timeout=s.llm_timeout_s,
        max_retries=s.llm_max_retries,
    )
