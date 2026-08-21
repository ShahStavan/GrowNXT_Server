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

from reporting.engine import ReportError, generate_report  # noqa: E402
from scripts import cli  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    p = argparse.ArgumentParser(description="Generate an institutional PDF equity report.")
    p.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. WIPRO")
    p.add_argument("--output-dir", type=Path,
                   help="Root of the per-ticker build workspace (default: output/).")
    p.add_argument("--reports-dir", type=Path,
                   help="Directory collecting every finished PDF (default: reports/).")
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
    for ticker in args.tickers:
        sym = ticker.upper()
        try:
            path = generate_report(sym, output_dir=args.output_dir,
                                   reports_dir=args.reports_dir,
                                   refresh=args.refresh, as_of=args.as_of)
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
