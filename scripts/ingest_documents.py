"""Command-line entry point for the document ingestion pipeline.

Examples:
    Ingest the latest annual report, transcripts and decks for one ticker::

        python -m scripts.ingest_documents WIPRO

    Three years of filings, transcripts only, with a fresh parse::

        python -m scripts.ingest_documents WIPRO --annual-reports 3 \\
            --doc-types concall_transcript --force parse

    Show what is already ingested without touching the network::

        python -m scripts.ingest_documents WIPRO --status

Google Python Style Guide Compliant.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import List, Optional

# Allow execution as a script from the project root as well as with -m.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import DATA_DIR  # noqa: E402
from ingestion.chunker import DEFAULT_ANNUAL_SKIP_SECTIONS  # noqa: E402
from ingestion.graph import IngestionPipeline  # noqa: E402
from ingestion.registry import DOC_TYPES, STAGES, DocumentRegistry  # noqa: E402


def _configure_logging(verbose: bool) -> None:
    """Sets up console logging.

    Args:
        verbose: Emit debug-level records.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            # Filings carry rupee signs and typographic dashes; a Windows
            # console defaults to cp1252 and would raise on the first one.
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_status(registry: DocumentRegistry) -> None:
    """Prints one row per catalogued document and its stage states."""
    rows = registry.status_table()
    if not rows:
        print("No documents catalogued for %s yet." % registry.ticker)
        return

    header = ["doc_id", "label"] + STAGES + ["chunks", "embed_model"]
    widths = [max(len(header[i]), 8) for i in range(len(header))]
    table = []
    for row in rows:
        values = [row["doc_id"], row["label"]]
        values += [str(row.get(stage, "-")) for stage in STAGES]
        values += [str(row.get("chunks", 0)), str(row.get("embed_model", ""))]
        table.append(values)
        for index, value in enumerate(values):
            widths[index] = max(widths[index], len(value))

    def line(values: List[str]) -> str:
        """Formats one padded row."""
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    print(line(header))
    print("  ".join("-" * width for width in widths))
    for values in table:
        print(line(values))

    counts = registry.counts()
    print("\n%s: %d documents; %s" % (
        registry.ticker, counts["documents"],
        ", ".join("%s=%d" % (stage, counts[stage]) for stage in STAGES),
    ))


def main(argv: Optional[List[str]] = None) -> int:
    """Parses arguments and runs the pipeline.

    Args:
        argv: Argument list, defaulting to sys.argv[1:].

    Returns:
        Process exit status: 0 on success, 1 when any document failed a stage.
    """
    parser = argparse.ArgumentParser(
        description="Ingest a company's filings into the document registry.",
    )
    parser.add_argument("ticker", help="Stock ticker symbol, e.g. WIPRO")
    parser.add_argument("--data-dir", default=None,
                        help="Root data directory (default: %s)" % DATA_DIR)
    parser.add_argument("--annual-reports", type=int, default=1,
                        help="Annual report years to request (default: 1)")
    parser.add_argument("--concall-years", type=int, default=1,
                        help="Years of concalls to request (default: 1)")
    parser.add_argument("--doc-types", nargs="*", choices=DOC_TYPES, default=None,
                        help="Document classes to ingest (default: all)")
    parser.add_argument("--embed-model", default=None,
                        help="Embedding model identifier (default: qwen3-embed)")
    parser.add_argument("--force", nargs="*",
                        choices=["download", "parse", "chunk", "embed", "prompt"],
                        default=None, help="Stages to run regardless of state")
    parser.add_argument("--no-prompts", action="store_true",
                        help="Skip building extraction prompts")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Cap pages parsed per document (smoke tests)")
    parser.add_argument("--skip-sections", nargs="*", default=None,
                        help="Annual-report sections to leave out of chunking. "
                             "Defaults to %s. Pass the flag with no values to "
                             "embed every section, including the audited "
                             "statements." % " ".join(DEFAULT_ANNUAL_SKIP_SECTIONS))
    parser.add_argument("--status", action="store_true",
                        help="Print the registry's state and exit")
    parser.add_argument("--json", action="store_true",
                        help="Print the run summary as JSON")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    data_dir = Path(args.data_dir) if args.data_dir else Path(DATA_DIR)

    if args.status:
        _print_status(DocumentRegistry(args.ticker, data_dir))
        return 0

    kwargs = {
        "data_dir": data_dir,
        "annual_reports": args.annual_reports,
        "concall_years": args.concall_years,
        "doc_types": args.doc_types,
        "force": args.force,
        "build_prompts": not args.no_prompts,
        "max_pages": args.max_pages,
        "skip_sections": args.skip_sections,
    }
    if args.embed_model:
        kwargs["embed_model"] = args.embed_model

    pipeline = IngestionPipeline(args.ticker, **kwargs)
    state = pipeline.run()

    if args.json:
        print(json.dumps({
            "ticker": state.get("ticker"),
            "company_name": state.get("company_name"),
            "planned": state.get("planned"),
            "actions": state.get("actions"),
            "errors": state.get("errors"),
            "counts": pipeline.registry.counts(),
        }, indent=2))
    else:
        print()
        _print_status(pipeline.registry)
        errors = state.get("errors") or []
        if errors:
            print("\n%d failure(s):" % len(errors))
            for item in errors:
                print("  %-28s %-9s %s" % (item["doc_id"], item["stage"], item["error"]))

    return 1 if state.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
