"""Hosted model invocation for GrowNXT Server.

All generation goes through one OpenAI-compatible chat-completions endpoint.
Model selection is managed server-side by the deployment, so this module sends
a prompt and returns the completion -- it does not name a model, and there is
no provider to choose.

Failures raise ``LLMError``. An earlier version fell back to returning the
prompt's own context when the endpoint was unreachable, which produced output
that read like analysis but was unprocessed source text; a caller cannot detect
that, so a raised error is the honest outcome.
"""

import logging
import os
from typing import Any, Dict, Final, Optional

from dotenv import load_dotenv
import requests

logger = logging.getLogger(__name__)

load_dotenv()

API_URL: Final[str] = os.getenv(
    "GROWNXT_LLM_API_URL", "https://grownxt-llm.vercel.app"
).rstrip("/")

# Long-form sections legitimately take over a minute to generate.
REQUEST_TIMEOUT: Final[int] = int(os.getenv("GROWNXT_LLM_TIMEOUT", "120"))


class LLMError(RuntimeError):
    """Raised when the hosted endpoint refuses, fails, or returns nothing."""


def _headers() -> Dict[str, str]:
    """Builds request headers, attaching the API key only when one is set."""
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("GROWNXT_LLM_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = "Bearer %s" % api_key
    return headers


def _completion_text(payload: Any) -> Optional[str]:
    """Extracts the assistant message from a chat-completions response body."""
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    return str(content).strip() if content else None


def generate_llm_response(prompt: str, context: str = "") -> str:
    """Generates a completion for ``prompt``, grounded in ``context``.

    Args:
        prompt: Task instructions.
        context: Ground-truth data the answer must be drawn from.

    Returns:
        str: The completion text.

    Raises:
        LLMError: If the request fails or the response carries no content.
    """
    full_prompt = "%s\n\nGround Truth Context Data:\n%s" % (prompt, context) if context else prompt

    try:
        response = requests.post(
            "%s/v1/chat/completions" % API_URL,
            json={"messages": [{"role": "user", "content": full_prompt}], "stream": False},
            headers=_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        content = _completion_text(response.json())
    except requests.RequestException as exc:
        raise LLMError("Hosted model request failed: %s" % exc) from exc
    except ValueError as exc:
        raise LLMError("Hosted model returned a malformed response body: %s" % exc) from exc

    if not content:
        raise LLMError("Hosted model returned an empty completion.")

    logger.debug("Hosted model returned %d characters.", len(content))
    return content
