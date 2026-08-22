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

from dataclasses import dataclass
import logging
from pathlib import Path

from core.config import OUTPUT_DIR, safe_ticker

logger = logging.getLogger(__name__)

# Sub-directories of output/<TICKER>/ owned by this package, and the suffix each
# one holds. Named once so no two stages can disagree.
DOCUMENTS_DIR: str = "documents"
EXTRACTED_DIR: str = "extracted"
FIGURES_DIR: str = "figures"

PDF_SUFFIX: str = ".pdf"
EXTRACTION_SUFFIX: str = ".json"
FIGURE_SUFFIX: str = ".png"


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
    def open(cls, ticker: str, data_dir: Path | None = None) -> "DocumentStore":
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

    def ensure(self) -> "DocumentStore":
        """Creates the directories this package writes into, and returns self."""
        for path in (self.documents, self.extracted, self.figures):
            path.mkdir(parents=True, exist_ok=True)
        return self

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
        name = "p%04d-%02d%s" % (max(0, page), max(0, index), FIGURE_SUFFIX)
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
            logger.warning("[%s] path %s is outside the store; recording it whole.",
                           self.ticker, path)
            return Path(path).as_posix()

    def resolve(self, relative_path: str) -> Path:
        """Returns the absolute path for a registry-recorded relative path."""
        return self.root / str(relative_path)
