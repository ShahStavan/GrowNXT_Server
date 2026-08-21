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
from typing import List, Optional, Sequence

# Allow execution as a script from the project root as well as with -m.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import OUTPUT_DIR  # noqa: E402
from ingestion.chunker import DEFAULT_ANNUAL_SKIP_SECTIONS  # noqa: E402
from ingestion.embedder import DEFAULT_EMBED_MODEL  # noqa: E402
from ingestion.graph import IngestionPipeline  # noqa: E402
from ingestion.registry import DOC_TYPES, STAGES, DocumentRegistry  # noqa: E402
from scripts import cli  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    p = argparse.ArgumentParser(
        description="Ingest a company's filings into the document registry.")
    p.add_argument("ticker", help="Stock ticker symbol, e.g. WIPRO")
    p.add_argument("--data-dir", help="Root artifact directory (default: %s)" % OUTPUT_DIR)
    p.add_argument("--annual-reports", type=int, default=1,
                   help="Annual report years to request (default: 1)")
    p.add_argument("--concall-years", type=int, default=1,
                   help="Years of concalls to request (default: 1)")
    p.add_argument("--doc-types", nargs="*", choices=DOC_TYPES,
                   help="Document classes to ingest (default: all)")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                   help="Embedding model identifier (default: %(default)s)")
    p.add_argument("--force", nargs="*", choices=STAGES,
                   help="Stages to run regardless of recorded state")
    p.add_argument("--no-prompts", action="store_true",
                   help="Skip building extraction prompts")
    p.add_argument("--max-pages", type=int,
                   help="Cap pages parsed per document (smoke tests)")
    p.add_argument("--skip-sections", nargs="*",
                   help="Annual-report sections to leave out of chunking. Defaults "
                        "to %s. Pass the flag with no values to embed every "
                        "section, including the audited statements."
                        % " ".join(DEFAULT_ANNUAL_SKIP_SECTIONS))
    p.add_argument("--status", action="store_true",
                   help="Print the registry's state and exit")
    p.add_argument("--json", action="store_true", help="Print the run summary as JSON")
    p.add_argument("--verbose", action="store_true", help="Debug logging")
    return p


def _grid(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Renders a padded text table, sized to its widest cell per column."""
    widths = [max(len(str(cell)) for cell in column)
              for column in zip(header, *rows)] if rows else [len(h) for h in header]
    widths = [max(width, 8) for width in widths]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells))

    return "\n".join([line(header), "  ".join("-" * w for w in widths)]
                     + [line(row) for row in rows])


def _print_status(registry: DocumentRegistry) -> None:
    """Prints one row per catalogued document and its stage states."""
    rows = registry.status_table()
    if not rows:
        print("No documents catalogued for %s yet." % registry.ticker)
        return

    header = ["doc_id", "label"] + list(STAGES) + ["chunks", "embed_model"]
    table = [[row["doc_id"], row["label"]]
             + [str(row.get(stage, "-")) for stage in STAGES]
             + [str(row.get("chunks", 0)), str(row.get("embed_model", ""))]
             for row in rows]
    print(_grid(header, table))

    counts = registry.counts()
    print("\n%s: %d documents; %s" % (
        registry.ticker, counts["documents"],
        ", ".join("%s=%d" % (stage, counts[stage]) for stage in STAGES)))


def main(argv: Optional[List[str]] = None) -> int:
    """Runs the pipeline for one ticker.

    Args:
        argv: Argument list, defaulting to sys.argv[1:].

    Returns:
        0 on success, 1 when any document failed a stage.
    """
    args = _parser().parse_args(argv)
    cli.setup(logging.DEBUG if args.verbose else logging.INFO,
              cli.TIMED, datefmt=cli.CLOCK)

    data_dir = Path(args.data_dir) if args.data_dir else Path(OUTPUT_DIR)
    if args.status:
        _print_status(DocumentRegistry(args.ticker, data_dir))
        return 0

    pipeline = IngestionPipeline(
        args.ticker,
        data_dir=data_dir,
        annual_reports=args.annual_reports,
        concall_years=args.concall_years,
        doc_types=args.doc_types,
        embed_model=args.embed_model,
        force=args.force,
        build_prompts=not args.no_prompts,
        max_pages=args.max_pages,
        skip_sections=args.skip_sections,
    )
    state = pipeline.run()
    errors = state.get("errors") or []

    if args.json:
        print(json.dumps({
            "ticker": state.get("ticker"),
            "company_name": state.get("company_name"),
            "planned": state.get("planned"),
            "actions": state.get("actions"),
            "errors": errors,
            "counts": pipeline.registry.counts(),
        }, indent=2))
    else:
        print()
        _print_status(pipeline.registry)
        if errors:
            print("\n%d failure(s):" % len(errors))
            for item in errors:
                print("  %-28s %-9s %s" % (item["doc_id"], item["stage"], item["error"]))

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
