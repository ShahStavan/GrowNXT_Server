"""Every path the document stages read or write.

One module owns the layout, so "where did that document go?" has one answer
regardless of which stage is asking, and nothing else builds a path by string
concatenation.

Layout, rooted at ``OUTPUT_DIR`` (overridden by ``GROWNXT_OUTPUT_DIR``)::

    output/<TICKER>/
    |-- registry.json                    per-document, per-stage ingestion state
    |-- documents/<doc_id>.pdf           the filing as published, never rewritten
    |-- extracted/<doc_id>.json          text, tables, figure manifest
    |-- figures/<doc_id>/p0142-03.png    page 142, third figure on the page
    `-- chunks/ vectors/ findings/       later stages, owned by their own modules

Two invariants make a re-run safe. A document's identity is its ``doc_id``,
derived from the period rather than the URL, so a re-ingest overwrites in place
instead of accumulating copies. And extraction never writes into ``documents/``,
so it cannot corrupt the source it is reading.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from core.config import OUTPUT_DIR, safe_ticker

logger = logging.getLogger(__name__)

# Sub-directories of output/<TICKER>/ owned by this package, and the suffix each
# one holds. Named once so no two stages can disagree.
DOCUMENTS_DIR: str = "documents"
EXTRACTED_DIR: str = "extracted"
FIGURES_DIR: str = "figures"
CHUNKS_DIR: str = "chunks"

PDF_SUFFIX: str = ".pdf"
EXTRACTION_SUFFIX: str = ".json"
FIGURE_SUFFIX: str = ".png"

# `Downloader._stream` writes to a hidden ``.<doc_id>-XXXX.part`` beside its
# destination and renames it only after the transfer validates, so a torn
# ``.pdf`` is impossible -- but a hard kill leaves the temporary file behind and
# nothing globs for it. Swept here rather than on download, because the sweep
# must not race a transfer that is still running: only files older than this are
# removed, which is why the age gate is not merely tidiness.
PART_SUFFIX: str = ".part"
PART_FILE_TTL_HOURS: float = 6.0


@dataclass(frozen=True)
class DocumentStore:
    """The on-disk home of one ticker's filings.

    Frozen because a store is an address, not a state: two stages handed the
    same store must resolve every path identically.

    Attributes:
        ticker: Directory-safe ticker symbol.
        root: The stock's directory, ``<data_dir>/<TICKER>``.

    """

    ticker: str
    root: Path

    @classmethod
    def open(cls, ticker: str, data_dir: Path | None = None) -> DocumentStore:
        """Returns the store for one ticker, without touching the filesystem.

        Args:
            ticker: Exchange symbol, in any case and with any punctuation.
            data_dir: Root artefact directory. Defaults to OUTPUT_DIR; tests
                pass a temporary directory.

        Returns:
            The store. Directories are created by `ensure`, so naming a path
            cannot leave empty directories behind.

        """
        symbol = safe_ticker(ticker)
        base = Path(data_dir) if data_dir is not None else OUTPUT_DIR
        return cls(ticker=symbol, root=Path(base) / symbol)

    # --- Directories ---------------------------------------------------------

    @property
    def documents(self) -> Path:
        """Directory holding the downloaded PDFs."""
        return self.root / DOCUMENTS_DIR

    @property
    def extracted(self) -> Path:
        """Directory holding one extraction JSON per document."""
        return self.root / EXTRACTED_DIR

    @property
    def figures(self) -> Path:
        """Directory holding one sub-directory of images per document."""
        return self.root / FIGURES_DIR

    @property
    def chunks(self) -> Path:
        """Directory holding one chunk-set JSON per document."""
        return self.root / CHUNKS_DIR

    def ensure(self) -> DocumentStore:
        """Creates the directories this package writes into, and returns self."""
        for path in (self.documents, self.extracted, self.figures, self.chunks):
            path.mkdir(parents=True, exist_ok=True)
        self.sweep_partials()
        return self

    def sweep_partials(self, ttl_hours: float = PART_FILE_TTL_HOURS) -> int:
        """Removes abandoned download temporaries, returning the bytes reclaimed.

        Args:
            ttl_hours: Minimum age before a temporary is presumed abandoned. A
                concurrent worker may be mid-transfer on this same ticker, so a
                fresh temporary is left alone.

        Returns:
            Bytes reclaimed. Never raises: an un-deletable temporary is a
            wasted megabyte, not a failed run.

        """
        cutoff = time.time() - max(0.0, ttl_hours) * 3600.0
        reclaimed = 0
        for path in self.documents.glob(f"*{PART_SUFFIX}"):
            try:
                stat = path.stat()
                if stat.st_mtime > cutoff:
                    continue
                path.unlink()
            except OSError:
                continue
            reclaimed += stat.st_size
            logger.debug(
                "[%s] removed abandoned download temporary %s (%d bytes)",
                self.ticker,
                path.name,
                stat.st_size,
            )
        return reclaimed

    # --- Files ---------------------------------------------------------------

    def pdf(self, doc_id: str) -> Path:
        """Returns the path of one filing's PDF."""
        return self.documents / (doc_id + PDF_SUFFIX)

    def extraction(self, doc_id: str) -> Path:
        """Returns the path of one filing's extraction JSON."""
        return self.extracted / (doc_id + EXTRACTION_SUFFIX)

    def figure_dir(self, doc_id: str) -> Path:
        """Returns the directory holding one filing's extracted images."""
        return self.figures / doc_id

    def chunk_file(self, doc_id: str) -> Path:
        """Returns the path of one filing's chunk-set JSON."""
        return self.chunks / (doc_id + EXTRACTION_SUFFIX)

    def figure(self, doc_id: str, page: int, index: int) -> Path:
        """Returns ``figures/<doc_id>/p<page>-<index>.png`` for one figure.

        Args:
            doc_id: Owning document.
            page: 1-based page the figure was cropped from.
            index: 1-based position of the figure on that page.

        Returns:
            The path, zero-padded so a directory listing sorts into document
            order -- what makes a figure directory reviewable by eye.

        """
        name = f"p{max(0, page):04d}-{max(0, index):02d}{FIGURE_SUFFIX}"
        return self.figure_dir(doc_id) / name

    # --- Registry paths ------------------------------------------------------

    def relative(self, path: Path) -> str:
        """Returns `path` relative to the stock directory, for the registry.

        Portable paths keep recorded state valid when the output directory is
        moved or mounted elsewhere. A path outside the store is recorded whole
        rather than forced: a silently wrong relative path is harder to diagnose
        than an absolute one.
        """
        try:
            return Path(path).resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            logger.warning(
                "[%s] path %s is outside the store; recording it whole.",
                self.ticker,
                path,
            )
            return Path(path).as_posix()

    def resolve(self, relative_path: str) -> Path:
        """Returns the absolute path for a registry-recorded relative path."""
        return self.root / str(relative_path)
