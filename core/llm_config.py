"""Hosted model invocation and SLM integration for GrowNXT Server.

Routes OpenAI-compatible chat completions directly to the upstream SLM endpoint
(https://llm.maqsoftware.net/v1 or configured reverse proxy) using direct requests,
with support for token-by-token streaming, configurable chat models (qwen-3.8-27b,
gemma-4-31b), and automatic sanitization of <think> reasoning tokens.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Generator
from typing import Any, Final

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# Upstream SLM Configuration loaded from environment (.env)
DEFAULT_BASE_URL: Final[str] = (
    (
        os.getenv("DEFAULT_BASE_URL")
        or os.getenv("GROWNXT_LLM_API_URL")
        or "https://llm.maqsoftware.net/v1"
    )
    .strip()
    .rstrip("/")
)

DEFAULT_CHAT_MODEL: Final[str] = os.getenv("DEFAULT_CHAT_MODEL", "qwen-3.8-27b").strip()
DEFAULT_CHAT_MODELS: Final[str] = os.getenv(
    "DEFAULT_CHAT_MODELS", "qwen-3.8-27b,gemma-4-31b"
).strip()
DEFAULT_EMBED_MODEL: Final[str] = os.getenv(
    "DEFAULT_EMBED_MODEL", "qwen3-embed"
).strip()
DEFAULT_RERANK_MODEL: Final[str] = os.getenv(
    "DEFAULT_RERANK_MODEL", "bge-reranker"
).strip()

# Configured API Base URL (defaults to upstream SLM endpoint from .env)
_raw_api_url: str = (
    os.getenv("GROWNXT_LLM_API_URL", DEFAULT_BASE_URL).strip().rstrip("/")
)
API_URL: Final[str] = (
    _raw_api_url if _raw_api_url.endswith("/v1") else f"{_raw_api_url}/v1"
)

# Active chat model selection
ACTIVE_MODEL: Final[str] = os.getenv("GROWNXT_LLM_MODEL", DEFAULT_CHAT_MODEL).strip()

# Request timeouts
REQUEST_TIMEOUT: Final[int] = int(os.getenv("GROWNXT_LLM_TIMEOUT", "180"))

# Regex to strip thinking tags from models that have thinking enabled by default
RE_THINKING_BLOCK = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
RE_UNCLOSED_THINKING = re.compile(r"<think>[\s\S]*$", re.IGNORECASE)


class LLMError(RuntimeError):
    """Raised when the hosted endpoint refuses, fails, or returns an error."""


def clean_thinking_tokens(text: str) -> str:
    """Strips internal <think>...</think> reasoning tokens from model outputs.

    Ensures raw chain-of-thought tokens are never leaked into user-facing
    documents, JSON findings, or compiled PDF reports.

    Args:
        text: Raw response string from the model.

    Returns:
        Cleaned string with thinking tokens removed.
    """
    if not text:
        return ""
    # Strip complete <think>...</think> blocks
    cleaned = RE_THINKING_BLOCK.sub("", text)
    # Strip unclosed <think> tag if output was truncated mid-reasoning
    cleaned = RE_UNCLOSED_THINKING.sub("", cleaned)
    return cleaned.strip()


def extract_thinking_and_content(text: str) -> tuple[str, str]:
    """Separates thinking reasoning trace from the final synthesized content.

    Args:
        text: Raw response string containing potential <think> tags.

    Returns:
        Tuple of (thinking_trace, clean_content).
    """
    if not text:
        return "", ""
    think_matches = RE_THINKING_BLOCK.findall(text)
    thinking_trace = "\n".join(
        m.replace("<think>", "").replace("</think>", "").strip() for m in think_matches
    )
    clean_content = clean_thinking_tokens(text)
    return thinking_trace, clean_content


def _headers() -> dict[str, str]:
    """Builds request headers, attaching the API key only when one is set."""
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("GROWNXT_LLM_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _completion_text(payload: Any) -> str | None:
    """Extracts and sanitizes the assistant message from a chat-completions response."""
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content:
        return clean_thinking_tokens(str(content))
    return None


def generate_llm_response(
    prompt: str,
    context: str = "",
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str:
    """Generates a completion for ``prompt``, grounded in ``context``.

    Args:
        prompt: Task instructions or user query.
        context: Ground-truth data context.
        model: Model identifier (defaults to ACTIVE_MODEL / qwen-3.8-27b).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        str: The sanitized completion text (with thinking tokens removed).

    Raises:
        LLMError: If the request fails or the response carries no content.
    """
    full_prompt = (
        f"{prompt}\n\nGround Truth Context Data:\n{context}" if context else prompt
    )
    selected_model = model or ACTIVE_MODEL

    payload: dict[str, Any] = {
        "model": selected_model,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    endpoint = f"{API_URL}/chat/completions"

    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers=_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        content = _completion_text(response.json())
    except requests.RequestException as exc:
        raise LLMError(f"SLM endpoint request failed ({endpoint}): {exc}") from exc
    except ValueError as exc:
        raise LLMError(f"SLM endpoint returned a malformed response: {exc}") from exc

    if not content:
        raise LLMError("SLM endpoint returned an empty completion.")

    logger.debug("SLM model '%s' returned %d characters.", selected_model, len(content))
    return content


def stream_llm_response(
    prompt: str,
    context: str = "",
    model: str | None = None,
    temperature: float = 0.2,
    strip_thinking: bool = True,
) -> Generator[str, None, None]:
    """Streams completion tokens in real-time with zero buffering.

    Args:
        prompt: Task instructions or user query.
        context: Ground-truth data context.
        model: Model identifier.
        temperature: Sampling temperature.
        strip_thinking: Whether to suppress streaming inside <think> tags.

    Yields:
        str: Individual text token chunks as they arrive from the SLM.
    """
    full_prompt = (
        f"{prompt}\n\nGround Truth Context Data:\n{context}" if context else prompt
    )
    selected_model = model or ACTIVE_MODEL

    payload: dict[str, Any] = {
        "model": selected_model,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": temperature,
        "stream": True,
    }

    endpoint = f"{API_URL}/chat/completions"

    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers=_headers(),
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise LLMError(f"SLM streaming request failed ({endpoint}): {exc}") from exc

    inside_thinking = False
    buffer = ""

    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line_str.startswith("data:"):
            continue
        data_content = line_str[5:].strip()
        if data_content == "[DONE]":
            break

        try:
            chunk_data = json.loads(data_content)
            choices = chunk_data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            token = delta.get("content") or ""

            if not token:
                continue

            if not strip_thinking:
                yield token
                continue

            # Filter thinking tags in stream
            buffer += token
            if "<think>" in buffer:
                inside_thinking = True
                buffer = buffer.split("<think>", 1)[0]
                if buffer:
                    yield buffer
                    buffer = ""

            if inside_thinking:
                if "</think>" in token:
                    inside_thinking = False
                    after_think = token.split("</think>", 1)[1]
                    if after_think:
                        yield after_think
                continue

            yield token
            buffer = ""
        except Exception:
            continue
