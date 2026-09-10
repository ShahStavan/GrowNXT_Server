"""Comprehensive extraction and chunking evaluation across WIPRO, HDFCBANK, TCS, and RELIANCE.

Workflow:
1. Discovers all corporate filing PDFs for WIPRO, HDFCBANK, TCS, RELIANCE.
2. Extracts layout elements (blocks, tables, figures) using Docling Extractor.
3. Chunks elements using the LlamaIndex PDF Elements Chunker.
4. Validates chunk quality, node relationships, table headers, and figure metadata.
5. Saves results into output/<TICKER>/chunks/<doc_id>.json.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from ingestion.chunker import (
    ELEMENT_FIGURE,
    ELEMENT_TABLE,
    ELEMENT_TEXT,
    chunk_document,
    write_chunk_cache,
)
from ingestion.documents.content import ExtractedDocument
from ingestion.documents.extract import Extractor
from ingestion.documents.storage import DocumentStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("evaluator")


def setup_stock_documents(ticker: str, root_dir: Path) -> DocumentStore:
    """Ensures stock directory layout and prepares PDF filings for extraction."""
    store = DocumentStore.open(ticker, data_dir=root_dir).ensure()
    ticker_dir = root_dir / ticker

    # Check for direct report PDF (e.g. HDFCBANK_report.pdf)
    direct_report = ticker_dir / f"{ticker}_report.pdf"
    if direct_report.exists():
        target_doc = store.pdf("annual_report_FY2026")
        if not target_doc.exists():
            logger.info(
                "[%s] Copying %s to store documents as annual_report_FY2026.pdf",
                ticker,
                direct_report.name,
            )
            shutil.copy2(direct_report, target_doc)

    return store


def evaluate_stock(
    ticker: str, root_dir: Path, max_pages_per_doc: int = 1000
) -> list[dict[str, Any]]:
    """Runs extraction (Docling) and chunking (LlamaIndex) for all documents of a stock."""
    store = setup_stock_documents(ticker, root_dir)
    extractor = Extractor(ocr=False, figures=True)
    results: list[dict[str, Any]] = []

    pdf_files = sorted(store.documents.glob("*.pdf"))
    if not pdf_files:
        logger.warning("[%s] No PDF documents found in %s", ticker, store.documents)
        return results

    for pdf_path in pdf_files:
        doc_id = pdf_path.stem
        if doc_id.startswith("_probe") or (
            doc_id == "annual_report_FY2026"
            and (store.documents / "annual_report_FY2026_financials.pdf").exists()
        ):
            continue

        doc_type = "annual_report"
        if "presentation" in doc_id.lower():
            doc_type = "concall_presentation"
        elif "transcript" in doc_id.lower():
            doc_type = "concall_transcript"

        label = f"{ticker} {doc_type.replace('_', ' ').title()}"

        logger.info("=" * 60)
        logger.info("[%s] Processing document: %s (%s)", ticker, doc_id, doc_type)

        # 1. Extraction Phase (Docling)
        try:
            extract_cache = store.extraction(doc_id)
            if extract_cache.exists():
                logger.info(
                    "[%s] Loading existing extraction cache: %s",
                    ticker,
                    extract_cache.name,
                )
                doc_extracted = ExtractedDocument.from_dict(
                    json.loads(extract_cache.read_text(encoding="utf-8"))
                )
            else:
                logger.info(
                    "[%s] Extracting layout with Docling (max_pages=%d)...",
                    ticker,
                    max_pages_per_doc,
                )
                start_ext = time.time()
                doc_extracted = extractor.run(
                    pdf=pdf_path,
                    store=store,
                    doc_id=doc_id,
                    doc_type=doc_type,
                    ticker=ticker,
                    label=label,
                    max_pages=max_pages_per_doc,
                    write=True,
                )
                logger.info(
                    "[%s] Extracted %s in %.2fs (%d blocks, %d tables, %d figures)",
                    ticker,
                    doc_id,
                    time.time() - start_ext,
                    len(doc_extracted.blocks),
                    len(doc_extracted.tables),
                    len(doc_extracted.figures),
                )

            # 2. Chunking Phase (LlamaIndex)
            logger.info("[%s] Chunking elements via LlamaIndex...", ticker)
            start_chk = time.time()
            chunk_set = chunk_document(doc_extracted, chunk_size=800, chunk_overlap=100)
            chk_time = time.time() - start_chk
        except Exception as exc:
            logger.error("[%s] Error processing %s: %s", ticker, doc_id, exc)
            continue

        # 3. Save ChunkSet to output/<TICKER>/chunks/<doc_id>.json
        chunk_path = store.root / "chunks" / f"{doc_id}.json"
        write_chunk_cache(chunk_set, chunk_path)
        logger.info(
            "[%s] Saved %d chunks to %s", ticker, chunk_set.n_chunks, chunk_path
        )

        # 4. Convert to native LlamaIndex nodes and verify graph integrity
        llama_nodes = chunk_set.to_nodes()

        elem_counts = chunk_set.element_counts()
        tbl_chunks = [c for c in chunk_set.chunks if c.element_type == ELEMENT_TABLE]
        fig_chunks = [c for c in chunk_set.chunks if c.element_type == ELEMENT_FIGURE]

        record = {
            "ticker": ticker,
            "doc_id": doc_id,
            "doc_type": doc_type,
            "source_pages": doc_extracted.n_source_pages,
            "extracted_blocks": len(doc_extracted.blocks),
            "extracted_tables": len(doc_extracted.tables),
            "extracted_figures": len(doc_extracted.figures),
            "total_chunks": chunk_set.n_chunks,
            "text_chunks": elem_counts.get(ELEMENT_TEXT, 0),
            "table_chunks": elem_counts.get(ELEMENT_TABLE, 0),
            "figure_chunks": elem_counts.get(ELEMENT_FIGURE, 0),
            "llama_nodes": len(llama_nodes),
            "total_chars": chunk_set.n_chars,
            "avg_chunk_chars": round(chunk_set.n_chars / max(1, chunk_set.n_chunks), 1),
            "chunk_time_sec": round(chk_time, 3),
            "table_header_preserved": all(
                c.metadata.get("header_rows", 0) > 0
                for c in tbl_chunks
                if c.table and c.table.header_rows > 0
            ),
            "figure_paths_valid": all(bool(c.figure_path) for c in fig_chunks),
        }
        results.append(record)

    return results


def main() -> None:
    output_dir = Path("output")
    target_tickers = ["WIPRO", "HDFCBANK", "TCS", "RELIANCE"]
    all_results: list[dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("GrowNXT Full Pipeline Evaluation: Docling Extraction + LlamaIndex Chunking")
    print("=" * 80 + "\n")

    for ticker in target_tickers:
        stock_results = evaluate_stock(ticker, output_dir, max_pages_per_doc=1000)
        all_results.extend(stock_results)

    print("\n" + "=" * 80)
    print("SUMMARY EVALUATION REPORT ACROSS ALL TICKERS")
    print("=" * 80)
    header = f"{'Ticker':<10} | {'Document':<32} | {'Blocks':<6} | {'Chunks':<6} | {'Txt':<4} | {'Tbl':<4} | {'Fig':<4} | {'Nodes':<5} | {'Chars':<7} | {'Status'}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        status = (
            "PASSED"
            if r["total_chunks"] > 0 and r["llama_nodes"] == r["total_chunks"]
            else "FAILED"
        )
        print(
            f"{r['ticker']:<10} | {r['doc_id']:<32} | {r['extracted_blocks']:<6} | {r['total_chunks']:<6} | {r['text_chunks']:<4} | {r['table_chunks']:<4} | {r['figure_chunks']:<4} | {r['llama_nodes']:<5} | {r['total_chars']:<7} | {status}"
        )

    print("=" * 80)
    print("All chunk files written to output/<TICKER>/chunks/")


if __name__ == "__main__":
    main()
