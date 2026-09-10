"""The Nifty 50 constituent universe, read from ``ticker_mapping.csv``.

Index membership is not derivable from anything the platform can query: the
Financial Data Collector service offers free-text company search only, and NSE
rebalances the index every March and September. The list therefore ships as
maintained data -- the repository's ``ticker_mapping.csv``, which already
records for each constituent the symbol the Collector service expects, the
Screener name, the sector, and the date the row was last verified.

This module is the one place that file is read for index membership, so the
batch embedding job and any later caller agree on what "all Nifty 50 stocks"
means.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import csv
import datetime as _dt
import logging
from dataclasses import dataclass
from pathlib import Path

from core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Shipped with the repository; ``load_nifty50(path=...)`` overrides it.
NIFTY50_MAPPING_PATH: Path = PROJECT_ROOT / "ticker_mapping.csv"

# Only rows whose lookup against the upstream services succeeded are members.
STATUS_OK: str = "ok"

# A rebalance has certainly happened once the file is this old.
STALE_AFTER_DAYS: int = 183


class Nifty50Error(RuntimeError):
    """Raised when the constituent file cannot be read or is malformed."""


@dataclass(frozen=True)
class Constituent:
    """One index member.

    Attributes:
        symbol: The exchange symbol the Collector service expects (``M&M``).
        name: Company name as published by Screener.
        sector: GICS-style sector label from the mapping file.
        status: Row status; only ``ok`` rows are index members.
        verified_at: ISO date the row was last checked, or ``""``.
    """

    symbol: str
    name: str = ""
    sector: str = ""
    status: str = STATUS_OK
    verified_at: str = ""


def load_nifty50(
    path: Path | None = None, include_unverified: bool = False
) -> list[Constituent]:
    """Reads the constituent list.

    Args:
        path: CSV to read; defaults to the repository's ``ticker_mapping.csv``.
        include_unverified: Keep rows whose ``status`` is not ``ok``. Off by
            default so a half-verified row cannot silently join a batch run.

    Returns:
        Constituents in file order, de-duplicated by upper-cased symbol.

    Raises:
        Nifty50Error: If the file is missing, has no ``ticker`` column, or is
            empty after filtering.
    """
    source = Path(path) if path is not None else NIFTY50_MAPPING_PATH
    if not source.exists():
        raise Nifty50Error(f"constituent file not found: {source}")

    members: list[Constituent] = []
    seen: set[str] = set()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "ticker" not in reader.fieldnames:
            raise Nifty50Error(f"{source} has no 'ticker' column")
        for row in reader:
            symbol = (row.get("ticker") or "").strip().upper()
            if not symbol or symbol in seen:
                continue
            status = (row.get("status") or STATUS_OK).strip().lower()
            if status != STATUS_OK and not include_unverified:
                logger.warning("[%s] skipped: status=%s", symbol, status)
                continue
            seen.add(symbol)
            members.append(
                Constituent(
                    symbol=symbol,
                    name=(
                        row.get("screener_name") or row.get("tickertape_name") or ""
                    ).strip(),
                    sector=(row.get("sector") or "").strip(),
                    status=status,
                    verified_at=(row.get("verified_at") or "").strip(),
                )
            )

    if not members:
        raise Nifty50Error(f"{source} lists no usable constituents")
    return members


def nifty50_symbols(path: Path | None = None) -> list[str]:
    """Returns the member symbols only, in file order."""
    return [member.symbol for member in load_nifty50(path)]


def verified_as_of(members: list[Constituent]) -> str:
    """Returns the most recent ``verified_at`` date across members, or ``""``."""
    dates = [m.verified_at for m in members if m.verified_at]
    return max(dates) if dates else ""


def is_stale(members: list[Constituent], today: _dt.date | None = None) -> bool:
    """Whether the list is old enough that an NSE rebalance has likely passed."""
    latest = verified_as_of(members)
    if not latest:
        return True
    try:
        checked = _dt.date.fromisoformat(latest[:10])
    except ValueError:
        return True
    now = today or _dt.datetime.now(_dt.UTC).date()
    return (now - checked).days > STALE_AFTER_DAYS
