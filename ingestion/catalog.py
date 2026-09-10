"""Document catalogue from the Financial Data Collector service.

``GET /api/v1/stocks/<symbol>/documents`` returns the Screener.in document
catalogue for a company: annual reports by financial year, and concall entries
that each carry a transcript URL and, when the issuer published one, a
presentation URL. This module turns that payload into a flat list of documents
with stable identifiers, which is what the registry needs to decide what has
already been ingested.

Two properties of the upstream payload shape the code:

* The response is a catalogue of *links*, not of content. A link changing is
  the only reliable signal that the issuer has published a new version of a
  document, so identifiers are derived from the period rather than from the
  URL, and the URL is carried alongside as the field that drives re-ingestion.
* Concall entries are not unique by date. An issuer routinely publishes two
  entries for one quarter -- the exchange filing and the copy on its own
  investor-relations site -- so identifiers are disambiguated by occurrence
  within a period rather than assumed unique.

Google Python Style Guide Compliant.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import requests

from core.config import FINANCIAL_DATA_COLLECTOR_BASE_URL, HTTP_HEADERS
from ingestion.fetcher import (
    DOC_TYPE_ANNUAL_REPORT,
    DOC_TYPE_PRESENTATION,
    DOC_TYPE_TRANSCRIPT,
    utc_now,
)

logger = logging.getLogger(__name__)

DOCUMENTS_ENDPOINT: str = "/api/v1/stocks/{sym}/documents"

# Month abbreviations used in concall period labels, e.g. "Jul 2026".
_MONTHS: dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_NON_ID_RE = re.compile(r"[^A-Za-z0-9]+")


class CatalogError(RuntimeError):
    """Raised when the catalogue cannot be retrieved for a ticker."""


@dataclass
class CatalogEntry:
    """One document offered by the catalogue.

    Attributes:
        doc_id: Stable identifier, unique within a ticker.
        doc_type: Document class.
        label: Human-readable period label.
        source_url: URL to download.
        period: Raw period fields as published upstream.

    """

    doc_id: str
    doc_type: str
    label: str
    source_url: str
    period: dict[str, Any] = field(default_factory=dict)


@dataclass
class Catalog:
    """A ticker's document catalogue.

    Attributes:
        ticker: Symbol as requested.
        company_name: Name as published by the catalogue.
        entries: Every document offered, in pipeline order.
        meta: Provenance and counts, stored in the registry for traceability.

    """

    ticker: str
    company_name: str = ""
    entries: list[CatalogEntry] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def of_type(self, doc_type: str) -> list[CatalogEntry]:
        """Returns the entries of one document class."""
        return [entry for entry in self.entries if entry.doc_type == doc_type]


def _slug(text: str) -> str:
    """Returns an identifier-safe form of a label."""
    return _NON_ID_RE.sub("_", (text or "").strip()).strip("_").upper()


def _period_sort_key(date: str, period: str) -> str:
    """Returns a sortable ``YYYY_MM`` key for a concall period.

    The catalogue's ``date`` field is already ``YYYY-MM`` for every entry seen
    in practice; ``period`` ("Jul 2026") is parsed as the fallback so that a
    change of upstream format degrades to a usable ordering rather than to none.
    """
    match = re.match(r"^(\d{4})-(\d{1,2})$", (date or "").strip())
    if match:
        return f"{match.group(1)}_{int(match.group(2)):02d}"
    parts = (period or "").replace(",", " ").split()
    month = 0
    year = 0
    for part in parts:
        key = part[:3].lower()
        if key in _MONTHS:
            month = _MONTHS[key]
        elif part.isdigit() and len(part) == 4:
            year = int(part)
    if year:
        return f"{year:04d}_{month:02d}"
    return _slug(date or period) or "UNDATED"


def _annual_entries(payload: dict[str, Any]) -> list[CatalogEntry]:
    """Builds entries for the annual reports in a catalogue payload."""
    entries: list[CatalogEntry] = []
    seen: dict[str, int] = {}
    for item in payload.get("annual_reports") or []:
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        year = _slug(str(item.get("financial_year") or "")) or "UNDATED"
        seen[year] = seen.get(year, 0) + 1
        suffix = "" if seen[year] == 1 else f"_{seen[year]}"
        entries.append(
            CatalogEntry(
                doc_id="annual_report_" + year + suffix,
                doc_type=DOC_TYPE_ANNUAL_REPORT,
                label=str(item.get("financial_year") or year),
                source_url=url,
                period={
                    "financial_year": item.get("financial_year"),
                    "source": item.get("source"),
                },
            )
        )
    return entries


def _concall_entries(payload: dict[str, Any]) -> list[CatalogEntry]:
    """Builds transcript and presentation entries from concall records.

    Both document classes come from the same upstream record, and either URL
    may be empty for a given quarter.
    """
    entries: list[CatalogEntry] = []
    seen: dict[tuple[str, str], int] = {}
    for item in payload.get("concalls") or []:
        date = str(item.get("date") or "")
        label = str(item.get("period") or date)
        key = _period_sort_key(date, label)
        for doc_type, field_name, prefix in (
            (DOC_TYPE_TRANSCRIPT, "transcript_url", "transcript_"),
            (DOC_TYPE_PRESENTATION, "ppt_url", "presentation_"),
        ):
            url = str(item.get(field_name) or "").strip()
            if not url:
                continue
            index = seen.get((doc_type, key), 0) + 1
            seen[(doc_type, key)] = index
            suffix = "" if index == 1 else f"_{index}"
            entries.append(
                CatalogEntry(
                    doc_id=prefix + key + suffix,
                    doc_type=doc_type,
                    label=label,
                    source_url=url,
                    period={"date": date, "period": label},
                )
            )
    return entries


def parse_catalog(ticker: str, payload: dict[str, Any]) -> Catalog:
    """Turns a catalogue payload into a Catalog.

    Args:
        ticker: Symbol the payload was requested for.
        payload: The ``data`` member of the service response.

    Returns:
        A Catalog with entries ordered annual reports first, then transcripts,
        then presentations, each newest first.

    """
    entries = _annual_entries(payload) + _concall_entries(payload)
    entries.sort(key=lambda entry: (entry.doc_type, entry.doc_id), reverse=True)
    return Catalog(
        ticker=ticker.upper(),
        company_name=str(payload.get("company_name") or ""),
        entries=entries,
        meta={
            "fetched_at": utc_now(),
            "source": payload.get("source"),
            "screener_url": payload.get("screener_url"),
            "screener_slug": payload.get("screener_slug"),
            "counts": payload.get("counts") or {},
            "annual_reports_requested": payload.get("annual_reports_requested"),
            "concall_years": payload.get("concall_years"),
            "entries_found": len(entries),
        },
    )


def fetch_catalog(
    ticker: str,
    annual_reports: int = 1,
    concall_years: int = 1,
    base_url: str = FINANCIAL_DATA_COLLECTOR_BASE_URL,
    timeout: int = 90,
    max_retries: int = 2,
) -> Catalog:
    """Fetches and parses one ticker's document catalogue.

    Args:
        ticker: Stock ticker symbol.
        annual_reports: Number of annual report years to request. The service
            defaults to one; earlier years cost an extra upstream fetch each.
        concall_years: Number of years of concalls to request.
        base_url: Service root without a trailing slash.
        timeout: Per-request timeout in seconds. The service scrapes on
            demand, so a cold request for several years is slow.
        max_retries: Additional attempts for transient failures.

    Returns:
        The parsed catalogue.

    Raises:
        CatalogError: If the catalogue cannot be retrieved or holds no
            documents, which means the ticker or the service is wrong.

    """
    # Percent-encode the symbol: an unescaped '&' in a ticker such as 'M&M'
    # would otherwise terminate the path and start a query string.
    symbol = quote(ticker.strip().lower(), safe="")
    url = base_url.rstrip("/") + DOCUMENTS_ENDPOINT.format(sym=symbol)
    params = {
        "annual_reports": int(annual_reports),
        "concall_years": int(concall_years),
    }

    payload: dict[str, Any] | None = None
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                url, params=params, headers=HTTP_HEADERS, timeout=timeout
            )
            if response.ok:
                body = response.json()
                payload = body.get("data") if isinstance(body, dict) else None
                if payload is None and isinstance(body, dict):
                    payload = body
                break
            last_error = f"HTTP {response.status_code}"
            if 400 <= response.status_code < 500 and response.status_code != 429:
                break
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
        logger.warning(
            "Catalogue request for %s failed (%s), attempt %d/%d",
            ticker,
            last_error,
            attempt + 1,
            max_retries + 1,
        )

    if not isinstance(payload, dict):
        raise CatalogError(
            f"Could not retrieve document catalogue for {ticker}: "
            f"{last_error or 'empty response'}"
        )

    catalog = parse_catalog(ticker, payload)
    if not catalog.entries:
        raise CatalogError(f"Catalogue for {ticker} lists no documents.")
    logger.info(
        "[%s] catalogue: %d documents (%s)",
        ticker.upper(),
        len(catalog.entries),
        ", ".join(
            f"{t}={len(catalog.of_type(t))}"
            for t in (
                DOC_TYPE_ANNUAL_REPORT,
                DOC_TYPE_TRANSCRIPT,
                DOC_TYPE_PRESENTATION,
            )
        ),
    )
    return catalog
