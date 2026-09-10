"""Stock discovery against the Financial Data Collector REST service.

One function. A search box is a read-only lookup, so there is no state worth
wrapping in a class -- only a reused connection pool.
"""

import logging
from typing import Any

import requests

from core.config import FINANCIAL_DATA_COLLECTOR_BASE_URL

log = logging.getLogger(__name__)

URL = FINANCIAL_DATA_COLLECTOR_BASE_URL + "/api/v1/stocks/search"
TIMEOUT = 15
MIN_CHARS = 3

# Module-level session: keep-alive across requests, since every search hits
# the same host.
_http = requests.Session()

Hit = dict[str, Any]


def find(q: str) -> list[Hit]:
    """Returns listed companies matching a name or symbol fragment.

    Every failure degrades to no matches: a type-ahead box must not raise a
    500 because an upstream lookup blinked, and the caller cannot act on the
    difference anyway.

    Args:
        q: Query text. Anything shorter than `MIN_CHARS` is not sent.

    Returns:
        Matching company records, in the order the service returns them.

    """
    q = (q or "").strip()
    if len(q) < MIN_CHARS:
        return []

    try:
        res = _http.get(URL, params={"q": q}, timeout=TIMEOUT)
        res.raise_for_status()
        body = res.json()
    except (requests.RequestException, ValueError) as exc:
        log.error("stock search failed for %r: %s", q, exc)
        return []

    if not isinstance(body, dict) or not body.get("success"):
        log.warning("stock search rejected %r: %s", q, body)
        return []

    hits = body.get("data") or []
    log.info("stock search %r matched %d", q, len(hits))
    return hits
