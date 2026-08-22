"""Command-line entry point for GrowNXT institutional PDF reports.

Usage:
    venv/Scripts/python.exe scripts/generate_report.py WIPRO
    venv/Scripts/python.exe scripts/generate_report.py WIPRO INFY --refresh
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import safe_ticker  # noqa: E402
from reporting.engine import ReportError, generate_report  # noqa: E402
from scripts import cli  # noqa: E402


def resolve_symbol(ticker_or_name: str) -> str:
    """Resolves an exchange symbol or company name to the canonical ticker symbol."""
    raw = (ticker_or_name or "").strip()
    if not raw:
        return ""
    upper = safe_ticker(raw)

    # 1. Check if local directory cache already exists
    try:
        from core.config import OUTPUT_DIR
        if (OUTPUT_DIR / upper / "api" / "summary.json").exists() or (OUTPUT_DIR / upper / "summary.json").exists():
            return upper
    except Exception:
        pass

    # 2. Lookup via stock discovery search API
    try:
        from api.search import find
        hits = find(raw)
        if hits:
            resolved = str(hits[0].get("ticker") or hits[0].get("sid") or upper).upper()
            if resolved != upper:
                logging.getLogger("generate_report").info("Resolved company '%s' -> ticker '%s'", raw, resolved)
            return safe_ticker(resolved)
    except Exception:
        pass

    return upper


def _parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    p = argparse.ArgumentParser(description="Generate an institutional PDF equity report.")
    p.add_argument("tickers", nargs="+", help="Ticker symbols or company names, e.g. WIPRO, Infosys")
    p.add_argument("--output-dir", type=Path,
                   help="Root of the per-ticker build workspace (default: output/).")
    p.add_argument("--keep-build", action="store_true",
                   help="Keep the chart SVGs and Typst source beside the PDF.")
    p.add_argument("--refresh", action="store_true",
                   help="Re-request collector data instead of using the local cache.")
    p.add_argument("--as-of", help="Display date for the header (default: today).")
    p.add_argument("--quiet", action="store_true", help="Log warnings and errors only.")
    return p


def main() -> int:
    """Generates one report per ticker.

    Returns:
        0 when every report was written, 1 otherwise.
    """
    args = _parser().parse_args()
    cli.setup(logging.WARNING if args.quiet else logging.INFO,
              cli.PLAIN, stream=sys.stderr)

    failed = []
    for ticker_input in args.tickers:
        sym = resolve_symbol(ticker_input)
        try:
            path = generate_report(sym, output_dir=args.output_dir,
                                   refresh=args.refresh, as_of=args.as_of,
                                   keep_build=args.keep_build)
        except Exception as exc:  # noqa: BLE001 - report and continue the batch
            # A ReportError is about this company's data; anything else is a
            # defect, and the label says which so a batch log stays readable.
            label = "FAILED" if isinstance(exc, ReportError) else "ERROR"
            print("%-12s %s: %s" % (sym, label, exc), file=sys.stderr)
            failed.append(sym)
        else:
            print("%-12s %s" % (sym, path))

    if failed:
        print("failed: %s" % ", ".join(failed), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
