"""System-wide configuration for GrowNXT Server.

Paths are derived from this file's own location, so a checkout works unchanged
on any machine, with environment overrides (``GROWNXT_OUTPUT_DIR``) for
containerised deployments.

Every artifact belonging to a stock -- the filings ingested for it, the parse
and vector caches, the extracted findings, and the typeset PDF report -- lives
under one directory per symbol: ``OUTPUT_DIR/<TICKER>/``. One root keeps a
company's evidence and its report together, and makes a stock's entire
footprint removable in a single step.
"""

import os
from pathlib import Path
from typing import Dict, Final, Tuple

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# Root of all per-stock artifacts. Override for container volumes.
_env_output_dir: str = os.getenv("GROWNXT_OUTPUT_DIR", "").strip()
OUTPUT_DIR: Final[Path] = (
    Path(_env_output_dir).resolve() if _env_output_dir else PROJECT_ROOT / "output"
)

# Every finished report, for every stock, in one directory. The build workspace
# stays per-stock under OUTPUT_DIR because Typst resolves chart paths relative
# to its source file and chart names repeat across tickers; only the compiled
# PDF is collected here, where a reader can find them all together.
_env_reports_dir: str = os.getenv("GROWNXT_REPORTS_DIR", "").strip()
REPORTS_DIR: Final[Path] = (
    Path(_env_reports_dir).resolve() if _env_reports_dir else PROJECT_ROOT / "reports"
)

# Ground-truth metric dictionary shipped with the repository.
MAPPING_FILE_PATH: Final[Path] = PROJECT_ROOT / "mapping.json"

# Live Financial Data Collector REST service.
FINANCIAL_DATA_COLLECTOR_BASE_URL: Final[str] = os.getenv(
    "FINANCIAL_DATA_SERVICE_URL",
    "https://financial-data-collector-qrxj.vercel.app",
).rstrip("/")

# Default outbound HTTP headers for the REST and document hosts.
HTTP_HEADERS: Final[Dict[str, str]] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/134.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}

# The hosted model endpoint carries its own default URL and treats its key as
# optional, so no variable has to be present for the server to start.
REQUIRED_ENV_VARS: Final[Tuple[str, ...]] = ()


def safe_ticker(ticker: str) -> str:
    """Returns a ticker normalised for use as a directory name.

    Indian symbols carry punctuation -- ampersands and hyphens are both common --
    and the symbol reaches the filesystem as a directory name, so every character
    that is not alphanumeric, a dash, or an underscore is folded to an
    underscore.
    """
    upper = (ticker or "").strip().upper()
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in upper) or "UNKNOWN"


def stock_dir(ticker: str, create: bool = False) -> Path:
    """Returns the build workspace for one stock: charts, caches, Typst source.

    Args:
        ticker: Exchange symbol, in any case and with any punctuation.
        create: Whether to create the directory when it is absent.

    Returns:
        Path: ``OUTPUT_DIR/<TICKER>``.
    """
    path = OUTPUT_DIR / safe_ticker(ticker)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def report_path(ticker: str, create_parent: bool = False) -> Path:
    """Returns the finished PDF's path for one stock.

    Args:
        ticker: Exchange symbol, in any case and with any punctuation.
        create_parent: Whether to create ``REPORTS_DIR`` when it is absent.

    Returns:
        Path: ``REPORTS_DIR/<TICKER>_report.pdf``.
    """
    if create_parent:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR / ("%s_report.pdf" % safe_ticker(ticker))
