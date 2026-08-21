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


def main() -> int:
    """Parses arguments and generates one report per ticker.

    Returns:
        Process exit code: 0 when every report was written, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Generate an institutional PDF equity report.",
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. WIPRO")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Root output directory (default: output/).",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-request collector data instead of using the local cache.",
    )
    parser.add_argument(
        "--as-of", default=None,
        help="Display date for the header and disclaimer (default: today).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Log warnings and errors only.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    failures = []
    for ticker in args.tickers:
        try:
            path = generate_report(
                ticker,
                output_dir=args.output_dir,
                refresh=args.refresh,
                as_of=args.as_of,
            )
            print("%-12s %s" % (ticker.upper(), path))
        except ReportError as exc:
            print("%-12s FAILED: %s" % (ticker.upper(), exc), file=sys.stderr)
            failures.append(ticker.upper())
        except Exception as exc:  # noqa: BLE001 - report and continue the batch
            print("%-12s ERROR: %s" % (ticker.upper(), exc), file=sys.stderr)
            failures.append(ticker.upper())

    if failures:
        print("failed: %s" % ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
