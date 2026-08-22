"""Typed client for the Financial Data Collector REST service.

Wraps the collector's endpoints for report generation, returning parsed
values with their absence made explicit -- a typeset report has to distinguish
a reported zero from a figure the collector does not carry.

Responses are cached on disk per ticker so that iterating on layout costs
no network traffic — a full report build re-reads one directory instead of
issuing fourteen requests.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from core.config import OUTPUT_DIR, safe_ticker

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL: str = "https://financial-data-collector-qrxj.vercel.app/api/v1"
DEFAULT_CACHE_DIR: Path = OUTPUT_DIR


# Endpoint suffixes keyed by the name used throughout the report code.
ENDPOINTS: Dict[str, str] = {
    "summary": "/stocks/{sym}/summary",
    "peers": "/stocks/{sym}/peers",
    "income_q": "/stocks/{sym}/income/quarterly",
    "income_a": "/stocks/{sym}/income/annual",
    "growth_q": "/stocks/{sym}/income/quarterly/growth",
    "growth_a": "/stocks/{sym}/income/annual/growth",
    "balance": "/stocks/{sym}/balancesheet",
    "balance_growth": "/stocks/{sym}/balancesheet/growth",
    "cashflow": "/stocks/{sym}/cashflow",
    "dupont": "/stocks/{sym}/dupont",
    "solvency": "/stocks/{sym}/solvency",
    "liquidity": "/stocks/{sym}/liquidity",
    "capital_efficiency": "/stocks/{sym}/capital-efficiency",
    "cagr": "/stocks/{sym}/cagr",
}


class CollectorError(RuntimeError):
    """Raised when a required endpoint cannot be retrieved."""


class CollectorClient:
    """Fetches and caches collector payloads for one or more tickers."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        cache_dir: Optional[Path] = None,
        timeout: int = 45,
        max_retries: int = 2,
        use_cache: bool = True,
    ) -> None:
        """Initializes the client.

        Args:
            base_url: Service root, including the version segment.
            cache_dir: Directory for cached JSON. Defaults to `.cache/api`.
            timeout: Per-request timeout in seconds.
            max_retries: Additional attempts for transient failures.
            use_cache: Read from cache when a payload is already present.
        """
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.use_cache = use_cache

    def _cache_path(self, ticker: str, name: str) -> Path:
        """Returns the cache path for one endpoint of one ticker."""
        sym = safe_ticker(ticker)
        # If cache_dir already addresses the stock's directory (e.g. output/WIPRO)
        if self.cache_dir.name == sym:
            return self.cache_dir / "api" / (name + ".json")
        return self.cache_dir / sym / "api" / (name + ".json")

    def _get(self, url: str) -> Optional[Any]:
        """GETs a URL and parses JSON, retrying transient failures."""
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(url, timeout=self.timeout)
                if response.ok:
                    return response.json()
                logger.warning(
                    "Collector returned %s for %s (attempt %d/%d)",
                    response.status_code, url, attempt + 1, self.max_retries + 1,
                )
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    return None
            except (requests.RequestException, ValueError) as exc:
                logger.warning("Collector request failed for %s: %s", url, exc)
            if attempt < self.max_retries:
                time.sleep(1.5 * (attempt + 1))
        return None

    def fetch(self, ticker: str, name: str, refresh: bool = False) -> Optional[Any]:
        """Fetches one endpoint payload, using the disk cache when allowed.

        Args:
            ticker: Stock ticker symbol.
            name: Key from `ENDPOINTS`.
            refresh: Bypass the cache and re-request.

        Returns:
            The payload's `data` member when present, else the whole
            document, or None when unavailable.
        """
        if name not in ENDPOINTS:
            raise KeyError("Unknown endpoint: " + name)

        path = self._cache_path(ticker, name)
        if self.use_cache and not refresh:
            # 1. Primary check: output/<TICKER>/api/<name>.json
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    return payload.get("data", payload) if isinstance(payload, dict) else payload
                except (OSError, ValueError) as exc:
                    logger.warning("Cache unreadable at %s: %s", path, exc)

            # 2. Legacy fallback check: .cache/api/<TICKER>/<name>.json
            legacy_path = Path(__file__).resolve().parent.parent / ".cache" / "api" / safe_ticker(ticker) / (name + ".json")
            if legacy_path.exists():
                try:
                    payload = json.loads(legacy_path.read_text(encoding="utf-8"))
                    try:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(payload), encoding="utf-8")
                    except Exception:
                        pass
                    return payload.get("data", payload) if isinstance(payload, dict) else payload
                except (OSError, ValueError) as exc:
                    logger.warning("Legacy cache unreadable at %s: %s", legacy_path, exc)

        # Percent-encode the symbol: an unescaped '&' in a ticker such as
        # 'M&M' would otherwise terminate the path and start a query string,
        # leaving the service to resolve 'm'. The collector decodes the
        # segment defensively on its side.
        symbol = quote(ticker.lower(), safe="")
        url = self.base_url + ENDPOINTS[name].format(sym=symbol)
        payload = self._get(url)
        if payload is None:
            return None

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not cache %s: %s", path, exc)

        return payload.get("data", payload) if isinstance(payload, dict) else payload

    def fetch_all(self, ticker: str, refresh: bool = False) -> Dict[str, Any]:
        """Fetches every endpoint for a ticker.

        Args:
            ticker: Stock ticker symbol.
            refresh: Bypass the cache for every endpoint.

        Returns:
            Mapping of endpoint name to payload. Missing endpoints map to
            None rather than being omitted, so downstream code can
            distinguish "absent" from "never requested".

        Raises:
            CollectorError: If no endpoint at all could be retrieved,
                which means the ticker or the service is wrong.
        """
        out: Dict[str, Any] = {}
        for name in ENDPOINTS:
            out[name] = self.fetch(ticker, name, refresh=refresh)
        if all(value is None for value in out.values()):
            raise CollectorError("No collector data available for " + ticker)

        missing: List[str] = [k for k, v in out.items() if v is None]
        if missing:
            logger.warning("[%s] endpoints unavailable: %s", ticker, ", ".join(missing))
        return out
