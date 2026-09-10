# Spec: Document Acquisition and Extraction

- **Status**: As-built reference
- **Author**: Claude, for @jenish.gajera
- **Date**: 2026-09-10
- **Covers**: `ingestion/documents/` — `storage.py`, `download.py`, `extract.py`,
  `content.py`, `sections.py`
- **Guarded by**: `scripts/verify_documents.py` (60 checks)

Receiving document for the design rationale that used to live in those modules' docstrings,
relocated by [`.claude/plans/code-style-refactor.md`](../plans/code-style-refactor.md) Phase 1.
Nothing here is new; it is the same reasoning, moved so the code reads as code.

---

## 1. Why this layer exists separately

Everything up to "a filing is text, tables and figures on disk". What a chunk is, and what a
section means, is decided downstream — this layer has no opinion. That boundary is what lets a
500-page annual report be re-chunked without being re-read, which matters because extraction is
the expensive stage: minutes of CPU per annual report.

A `doc_id` is derived from the period rather than the URL, so a re-ingest overwrites in place
instead of accumulating copies. Extraction never writes into `documents/`, so it cannot corrupt
the source it is reading.

---

## 2. `download.py` — four host behaviours, observed rather than guessed

An ingest of one company pulls four to a dozen PDFs from unrelated hosts: an exchange attachment
handler, the issuer's IR CDN, sometimes a registrar. Any one can be slow or down, so downloads
run on a small thread pool and results are consumed **as they land** — a 40 MB annual report
behind a slow exchange host must not hold up six transcripts already available.

**These four are load-bearing. Each was observed in production, and "simplifying" any of them
reintroduces a silent failure.**

1. **Exchange hosts refuse requests without a browser user agent, and want a referer from their
   own site.** The referer is derived *per request* from the URL being fetched, because the
   catalogue mixes hosts freely. One global referer does not work.
2. **IR sites answer a moved document with an HTML error page and HTTP 200.** A response is
   accepted only once its bytes parse as a PDF. Without that check an error page gets filed as a
   filing, and the failure surfaces much later as an empty extraction.
3. **A 404 is not a 503.** Permanent statuses are not retried; transient ones back off
   exponentially, honouring `Retry-After` when the host sends it.
4. **Hosts rate-limit per connection**, so each worker thread keeps its own `Session`.
   `requests` does not promise to serialise a shared one.

**Atomicity.** Bytes stream to a temporary file *inside* the destination directory and are
renamed into place, so an interrupted run leaves no truncated PDF for a later run to mistake for
a complete one — and the rename is atomic because it never crosses a filesystem. The hash is
computed from the stream, so a 40 MB filing is read once, not twice.

A hard kill still leaves a `.part` behind and nothing globs for it, which is why
`DocumentStore.sweep_partials` exists and why its age gate is correctness rather than tidiness:
a concurrent worker may be mid-transfer on the same ticker.

---

## 3. `extract.py` — what Docling supplies

Replaced a hand-tuned pdfplumber stack — word boxes, a recursive XY-cut for multi-column pages,
point thresholds for gutters and cells, a keyword classifier over running heads — all of which
reconstructed from geometry what a layout model reports directly, on thresholds calibrated
against a handful of reports.

- **Reading order from a layout model**, so a three-column page or a two-page spread comes out in
  the order a person reads it.
- **Table structure from TableFormer**, a grid with header cells marked.
- **A heading hierarchy**, so every block carries the trail it sits under — the document's own
  statement of its structure, and what the chunker sections on.
- **Figures as images, and OCR for pages that are pictures of text.** An investor deck is mostly
  charts and a scanned filing is entirely one; both used to yield nothing at all, silently.

**Operational**: model weights download once, on the first conversion, into Docling's cache
(`~/.cache/docling` unless `DOCLING_ARTIFACTS_PATH` says otherwise).

**Cache key.** `extract_version()` carries every setting that changes the extracted text — the
table mode (`+fast-tables`) and the page filter (`+fin-pages`), in that fixed order. The page
range enters `Extractor._cache_key` too, so a full extraction can never satisfy a filtered
request or the reverse. **Device and thread count must never enter either**, or every machine
would invalidate the last one's cache. `verify_style.py:check_frozen` pins these literals.

---

## 4. `content.py` — three decisions that shape retrieval

- **A block carries its heading trail** (`Block.path`, outermost first), so a chunker can say
  "this paragraph is inside Management Discussion, under Segment Review" from the document's own
  structure rather than guessing from keywords.
- **A table stays a table.** Header rows are kept apart from the body so the header can be
  re-attached to every fragment the table is split into. A row reading
  `Revenue from operations 926,163 790,935` without its header is not evidence — nothing in it
  says which column is which year, and that is the commonest way a financial table becomes
  unusable downstream.
- **A figure is a file plus what is known about it**: the image beside the document, its caption,
  and the classifier verdict, so a later vision pass can tell a bar chart from a logo.

Every dataclass round-trips through `to_dict`/`from_dict`: this is a cache format as much as an
in-memory model.

---

## 5. `sections.py` — the page filter

The financial content of an annual report is a **contiguous tail**, so the filter only has to
find where that tail starts — a pypdf text pass at ~60 ms/page against Docling's seconds.

**Two rules, both guarded by `check_page_filter`:**

- **Fail toward keeping pages.** Every uncertain outcome — no anchor, an unreadable or scanned
  PDF, a document under 60 pages, an anchor on page 1 — returns `None`, meaning convert
  everything. A dropped page is unsearchable forever; a spare one costs seconds.
- **An anchor must be the statements, not a phrase near them.** Page 191 of the ADANIENT FY2026
  filing opens "INDEPENDENT AUDITOR'S **CERTIFICATE** ON COMPLIANCE WITH THE CORPORATE
  GOVERNANCE REQUIREMENTS". An anchor matching bare `independent auditor` fires there and drags
  in 31 pages of the governance report the filter exists to drop, so `ANCHORS` requires the word
  "report".

Measured on ADANIENT FY2026: financial section at pages 222–396, dropping 221 of 396.

> **Note for the vectorless rebuild.** This module keeps the financial tail — the half that
> duplicates the REST API. The qualitative pipeline needs the *opposite* polarity; see
> [`vectorless-qualitative-rag.md`](vectorless-qualitative-rag.md) §2.1 and §4.3.
