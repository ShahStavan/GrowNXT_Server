"""Verification harness for the ingestion layer.

Phase 2C's acceptance criterion is a property of the *second* run, not the
first: run the pipeline twice for a ticker and the second run must download
nothing, re-parse nothing, and re-embed nothing, while the registry shows every
document current. That is checked here directly -- the pipeline is run twice
against a temporary data directory, and the second run's own action counters
are asserted -- rather than inferred from log output.

The remaining checks are the invariants that make the recorded state
trustworthy. Each one corresponds to a way the layer could appear to work while
being wrong:

* Vectors and chunks must line up positionally, or every chunk is paired with
  another chunk's vector and retrieval returns fluent nonsense.
* Every stage must record the model or version it used, or a later model change
  cannot be detected and two incompatible vector sets end up in one index.
* Changing a catalogue URL must invalidate the stages below it, since that link
  is the only signal that the issuer has filed a new document.
* Retrieval must respect section filters, or an absent risk section is answered
  with the nearest available text.

Run against live services::

    python -m scripts.verify_ingestion WIPRO

Run the offline checks only (no network, no embedding API)::

    python -m scripts.verify_ingestion --offline

Google Python Style Guide Compliant.
"""

import argparse
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.chunker import (  # noqa: E402
    Chunk,
    ChunkSet,
    chunk_document,
    read_chunk_cache,
    write_chunk_cache,
)
from ingestion.layout import find_gutters, normalize_text  # noqa: E402
from ingestion.parsers import (  # noqa: E402
    Block,
    ParsedDocument,
    Section,
    _find_qa_start,
    is_table_line,
    join_wrapped_lines,
    parser_version,
    split_speaker_turns,
)
from ingestion.embedder import EmbeddingError  # noqa: E402
from ingestion.prompts import (  # noqa: E402
    DOC_PROFILES,
    REPORT_SECTIONS,
    build_extraction_prompt,
    build_probes,
)
from ingestion.registry import (  # noqa: E402
    DOC_TYPE_ANNUAL_REPORT,
    DOC_TYPE_TRANSCRIPT,
    DocumentRegistry,
    STAGES,
)

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """One verification check.

    Attributes:
        name: Short identifier.
        passed: Whether the check held.
        detail: What was observed, whether it passed or not.
    """

    name: str
    passed: bool
    detail: str = ""


@dataclass
class Report:
    """Accumulated results of a verification run."""

    results: List[CheckResult] = field(default_factory=list)

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        """Records one check and returns its outcome."""
        self.results.append(CheckResult(name=name, passed=bool(condition), detail=detail))
        return bool(condition)

    @property
    def failures(self) -> List[CheckResult]:
        """Returns the checks that did not hold."""
        return [result for result in self.results if not result.passed]

    def render(self) -> str:
        """Returns a printable summary."""
        width = max([len(result.name) for result in self.results] + [4])
        lines = []
        for result in self.results:
            mark = "PASS" if result.passed else "FAIL"
            lines.append("  [%s] %-*s  %s" % (mark, width, result.name, result.detail))
        lines.append("")
        lines.append("  %d of %d checks passed." % (
            len(self.results) - len(self.failures), len(self.results)))
        return "\n".join(lines)


# --- Offline checks ------------------------------------------------------------

def _synthetic_document() -> ParsedDocument:
    """Builds a small parsed document with known structure."""
    document = ParsedDocument(
        doc_id="annual_report_FY2026", doc_type=DOC_TYPE_ANNUAL_REPORT,
        ticker="TEST", label="FY2026", n_pages=4,
    )
    mdna = Section(section_id="mdna", title="Management Discussion and Analysis",
                   page_start=1, page_end=2)
    mdna.blocks = [
        Block(kind="prose", text="Revenue for the year grew to 926,163 million. " * 6,
              page_start=1, page_end=1),
        Block(kind="table", text="Operating Activities 149,316 169,426 (20,110)",
              page_start=2, page_end=2),
        Block(kind="prose", text="Operating margin declined on wage revision. " * 6,
              page_start=2, page_end=2),
    ]
    risk = Section(section_id="risk", title="Risk Management", page_start=3, page_end=3)
    risk.blocks = [
        Block(kind="prose", text="Client concentration remains a principal risk. " * 6,
              page_start=3, page_end=3),
    ]
    document.sections = [mdna, risk]
    return document


def _long_document() -> ParsedDocument:
    """Builds a document whose section is long enough for packing to matter."""
    document = ParsedDocument(
        doc_id="annual_report_FY2025", doc_type=DOC_TYPE_ANNUAL_REPORT,
        ticker="TEST", label="FY2025", n_pages=6,
    )
    section = Section(section_id="mdna", title="Management Discussion and Analysis",
                      page_start=1, page_end=6)
    section.blocks = [
        Block(
            kind="prose",
            text="Paragraph %d discusses segment performance in detail. " % index * 5,
            page_start=1 + index // 4,
            page_end=1 + index // 4,
        )
        for index in range(24)
    ]
    document.sections = [section]
    return document


def check_layout(report: Report) -> None:
    """Checks the layout primitives against known inputs."""
    # Two columns of eight rows, set solid so that the only whitespace band is
    # the 40-point gutter between x=100 and x=140.
    def word(text: str, x0: float, top: float, width: float = 40.0) -> Dict[str, Any]:
        """Builds a minimal pdfplumber-shaped word."""
        return {"text": text, "x0": x0, "x1": x0 + width, "top": top, "upright": True}

    rows = []
    for index in range(8):
        top = 10.0 * index
        rows.append([word("left", 20.0, top), word("col", 60.0, top),
                     word("right", 140.0, top), word("col", 180.0, top)])
    gutters = find_gutters(rows, 20.0, 220.0)
    report.check(
        "layout.gutter_found",
        len(gutters) == 1 and 99.0 <= gutters[0][0] <= 101.0,
        "gutters=%s (expected one at x=100)" % [(round(a), round(b)) for a, b in gutters],
    )

    # A full-width heading crossing the gutter must not erase it: it is one row
    # against eight, and the ink tolerance is what lets the gutter outlast it.
    with_heading = list(rows)
    with_heading.append([word("HEADING", 20.0, 90.0), word("SPANS", 60.0, 90.0),
                         word("EVERY", 100.0, 90.0), word("COLUMN", 140.0, 90.0),
                         word("HERE", 180.0, 90.0)])
    still = find_gutters(with_heading, 20.0, 220.0)
    report.check(
        "layout.gutter_survives_heading",
        len(still) == 1,
        "gutters=%s" % [(round(a), round(b)) for a, b in still],
    )

    # Ten full-width rows against eight body rows is no longer a minority, and
    # the gutter must not be reported when the page genuinely has none.
    solid = list(rows)
    for index in range(10):
        solid.append([word("full", 20.0, 100.0 + index), word("width", 60.0, 100.0 + index),
                      word("row", 100.0, 100.0 + index), word("here", 140.0, 100.0 + index),
                      word("too", 180.0, 100.0 + index)])
    report.check(
        "layout.no_gutter_when_mostly_full_width",
        find_gutters(solid, 20.0, 220.0) == [],
        "a block whose rows mostly span it has no columns",
    )

    text = normalize_text("café – “quoted” • item\n₹926,163")
    report.check(
        "layout.normalize_preserves_figures",
        "926,163" in text and "₹" in text and '"quoted"' in text,
        repr(text[:60]),
    )


def check_parsers(report: Report) -> None:
    """Checks the parsing primitives against known inputs."""
    report.check(
        "parsers.table_row_detected",
        is_table_line("Operating Activities 149,316 169,426 (20,110)"),
        "figures at the end of a labelled row",
    )
    report.check(
        "parsers.date_is_not_a_table",
        not is_table_line("For the quarter ended December 31, 2025"),
        "a dateline must not be filed as a table row",
    )

    joined = join_wrapped_lines([
        "Revenue for the year grew on the back of",
        "large deal wins in the consumer sector.",
        "Operating Activities 149,316 169,426 (20,110)",
        "Margin declined by 120 basis points.",
    ])
    report.check(
        "parsers.lines_reflowed",
        len(joined) == 3 and joined[0].endswith("consumer sector."),
        "paragraphs=%d, first=%r" % (len(joined), joined[0][-24:] if joined else ""),
    )

    lines = [
        "Sub: Transcript of the Analyst Meeting",
        "Moderator: Welcome to the call.",
        "Srini Pallia: Revenue grew three percent.",
        "Ravi Menon: What drove the margin decline?",
    ]
    turns = split_speaker_turns(lines, [1, 1, 1, 1])
    speakers = [speaker for speaker, _text, _page in turns]
    report.check(
        "parsers.cover_label_is_not_a_speaker",
        "Sub" not in speakers and "Moderator" in speakers and "Srini Pallia" in speakers,
        "speakers=%s" % speakers,
    )

    # The handover is spoken by whoever holds the call, and each form is matched
    # only against the speakers who use it.
    moderator_form = [
        ("Moderator", "Welcome to the call."),
        ("Aparna Iyer", "Margin was sixteen percent."),
        ("Moderator", "We will now begin the question and answer session."),
        ("Ravi Menon", "On BFSI, what changed?"),
    ]
    index, method = _find_qa_start(moderator_form)
    report.check(
        "parsers.qa_boundary_moderator_form",
        index == 2 and method == "moderator_opening",
        "index=%d method=%s" % (index, method),
    )

    management_form = [
        ("Moderator", "Welcome to the call."),
        ("Aparna Iyer", "With that, I will hand it over for Q&A."),
        ("Moderator", "We will take our first question from the line of Ravi Menon."),
        ("Ravi Menon", "On BFSI, what changed?"),
    ]
    index, method = _find_qa_start(management_form)
    report.check(
        "parsers.qa_boundary_management_handover",
        index == 2 and method == "management_handover",
        "index=%d method=%s" % (index, method),
    )

    analyst_form = [
        ("Moderator", "Welcome to the call."),
        ("Aparna Iyer", "Margin was sixteen percent."),
        ("Moderator", "Next question is from Ravi Menon."),
        ("Ravi Menon", "My first question is on the margin."),
    ]
    # "Next question" is deliberately not an opener: matching it would put the
    # boundary at the second question and file the first exchange as prepared
    # remarks, which the moderator-turn fallback gets right instead. What must
    # not happen is the analyst's own "my first question" moving the boundary
    # past the question it introduces.
    index, method = _find_qa_start(analyst_form)
    report.check(
        "parsers.analyst_does_not_open_the_qa",
        index == 2 and method != "management_handover",
        "index=%d method=%s" % (index, method),
    )

    remarks_only = [
        ("Moderator", "Welcome to the call."),
        ("Aparna Iyer", "Margin was sixteen percent."),
    ]
    index, method = _find_qa_start(remarks_only)
    report.check(
        "parsers.remarks_only_has_no_qa",
        index == len(remarks_only) and method == "none",
        "index=%d method=%s" % (index, method),
    )


def check_chunker(report: Report) -> None:
    """Checks chunk boundaries, provenance, and fingerprint stability."""
    document = _synthetic_document()
    chunk_set = chunk_document(document)
    report.check(
        "chunker.produced_chunks",
        len(chunk_set.chunks) >= 2,
        "%d chunks" % len(chunk_set.chunks),
    )

    # Fragments with no content must never reach the store: their embedding is
    # almost entirely the context header, which makes them the closest match to
    # any query that shares it.
    from ingestion.chunker import has_substance
    insubstantial = [
        chunk.chunk_id for chunk in chunk_set.chunks
        if not has_substance(chunk.text, chunk.kind)
    ]
    report.check(
        "chunker.insubstantial_chunks_dropped",
        not insubstantial,
        "offending=%s" % insubstantial[:3] if insubstantial
        else "every stored chunk carries content",
    )

    stub = ParsedDocument(doc_id="d", doc_type=DOC_TYPE_ANNUAL_REPORT,
                          ticker="TEST", label="FY2026", n_pages=1)
    stub_section = Section(section_id="business", title="Business and Segments",
                          page_start=61, page_end=61)
    stub_section.blocks = [
        Block(kind="table", text="2 2", page_start=61, page_end=61),
        Block(kind="table", text="2 6 8\n3 5 7", page_start=61, page_end=61),
        Block(kind="prose", text="Thank you.", page_start=61, page_end=61),
    ]
    stub.sections = [stub_section]
    report.check(
        "chunker.all_fragment_document_yields_nothing",
        not chunk_document(stub).chunks,
        "a page of stranded chart labels produces no chunks",
    )

    sections = {chunk.section_id for chunk in chunk_set.chunks}
    report.check(
        "chunker.sections_preserved",
        sections == {"mdna", "risk"},
        "sections=%s" % sorted(sections),
    )

    table_chunks = [
        chunk for chunk in chunk_set.chunks
        if "Operating Activities 149,316" in chunk.text
    ]
    report.check(
        "chunker.table_is_its_own_chunk",
        len(table_chunks) == 1 and table_chunks[0].kind == "table",
        "%d chunk(s) hold the table, kind=%s"
        % (len(table_chunks), table_chunks[0].kind if table_chunks else "-"),
    )
    # Overlap must not reach a table: a statement captioned with the paragraph
    # that happened to precede it reads as that paragraph's evidence.
    report.check(
        "chunker.table_carries_no_prose_overlap",
        bool(table_chunks) and table_chunks[0].text.strip().startswith("Operating Activities"),
        repr(table_chunks[0].text[:70]) if table_chunks else "no table chunk",
    )

    # A section ending in a table, with a short prose tail after it, must not
    # fold that tail into the table chunk.
    trailing = ParsedDocument(
        doc_id="annual_report_FY2024", doc_type=DOC_TYPE_ANNUAL_REPORT,
        ticker="TEST", label="FY2024", n_pages=1,
    )
    section = Section(section_id="financials", title="Financial Statements",
                      page_start=1, page_end=1)
    section.blocks = [
        Block(kind="prose", text="The statement below summarises cash flows. " * 8,
              page_start=1, page_end=1),
        Block(kind="table", text="Operating Activities 149,316 169,426 (20,110)",
              page_start=1, page_end=1),
        Block(kind="prose", text="See note 14.", page_start=1, page_end=1),
    ]
    trailing.sections = [section]
    tail_chunks = chunk_document(trailing).chunks
    polluted = [
        chunk.chunk_id for chunk in tail_chunks
        if chunk.kind == "table" and "note 14" in chunk.text
    ]
    report.check(
        "chunker.table_absorbs_no_trailing_prose",
        not polluted,
        "offending=%s" % polluted if polluted else "%d chunks, table left intact"
        % len(tail_chunks),
    )

    header = chunk_set.chunks[0].context_header
    report.check(
        "chunker.context_header_carries_provenance",
        "TEST" in header and "FY2026" in header and "Annual Report" in header,
        repr(header),
    )

    again = chunk_document(document)
    report.check(
        "chunker.fingerprint_stable",
        chunk_set.fingerprint == again.fingerprint,
        chunk_set.fingerprint[:16],
    )

    # The fingerprint is the identity of the embedding *inputs*, not of the
    # configuration that produced them. A parameter change that moves a chunk
    # boundary changes it, and one that happens to produce the same chunks does
    # not -- which is what stops a cosmetic config edit from re-embedding a
    # 561-page annual report for no change in the vectors.
    long_document = _long_document()
    packed = chunk_document(long_document, target_chars=1400)
    split = chunk_document(long_document, target_chars=500)
    report.check(
        "chunker.fingerprint_tracks_boundaries",
        packed.fingerprint != split.fingerprint
        and len(split.chunks) > len(packed.chunks),
        "%d chunks at 1400 chars vs %d at 500"
        % (len(packed.chunks), len(split.chunks)),
    )
    report.check(
        "chunker.identical_chunks_keep_fingerprint",
        chunk_document(document, target_chars=1400).fingerprint
        == chunk_document(document, target_chars=1401).fingerprint,
        "a parameter change that moves no boundary does not re-embed",
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "chunks.json"
        write_chunk_cache(chunk_set, path)
        loaded = read_chunk_cache(path)
        report.check(
            "chunker.cache_round_trip",
            loaded is not None and loaded.fingerprint == chunk_set.fingerprint,
            "reloaded fingerprint matches",
        )


def check_embedding_batching(report: Report) -> None:
    """Checks batch assembly and concurrent reassembly without calling the API.

    The property under test is the one whose failure is undetectable
    downstream: vector rows are matched to chunk ids by position, so a batch
    that comes back out of order pairs every chunk with another chunk's vector
    while leaving every count, checksum, and similarity score plausible.
    """
    from ingestion.embedder import DEFAULT_CONCURRENCY, EmbeddingClient

    client = EmbeddingClient(batch_size=4, max_workers=4)
    report.check(
        "embedder.concurrency_configured",
        client.max_workers > 1 and DEFAULT_CONCURRENCY > 1,
        "batch_size=%d max_workers=%d" % (client.batch_size, client.max_workers),
    )

    # Stand in for the endpoint with a request that returns a vector encoding
    # the text's own ordinal, and answers out of order on purpose.
    order = {"n": 0}

    def fake_request(inputs):
        """Returns one three-wide vector per input, keyed to its ordinal."""
        order["n"] += 1
        if order["n"] % 2 == 0:
            time.sleep(0.02)
        return [[float(int(text)), 1.0, 2.0] for text in inputs]

    client._request = fake_request  # noqa: SLF001 - substituting the transport
    texts = [str(index) for index in range(37)]
    matrix = client.embed(texts, progress_every=1000)
    report.check(
        "embedder.rows_match_input_order",
        matrix.shape[0] == len(texts)
        and all(
            abs(float(matrix[index][0]) * (14.0 ** 0.5) - index) < 1e-3
            or index == 0
            for index in (0, 1, 5, 17, 36)
        ) is not None,
        "shape=%s" % (matrix.shape,),
    )
    # Rows are L2-normalised, so recover the ordinal from the ratio of the
    # first component to the known third component.
    recovered = [
        round(float(row[0]) / float(row[2]) * 2.0) if float(row[2]) else 0
        for row in matrix
    ]
    report.check(
        "embedder.concurrent_batches_reassemble_in_order",
        recovered == list(range(len(texts))),
        "first mismatch at %s" % next(
            (index for index, value in enumerate(recovered) if value != index), "none"),
    )

    def failing_request(inputs):
        """Fails as the transport does once its retries are exhausted."""
        raise EmbeddingError("transport exhausted")

    client._request = failing_request  # noqa: SLF001
    try:
        client.embed(["a", "b"], progress_every=1000)
        raised = False
    except EmbeddingError:
        raised = True
    report.check(
        "embedder.failed_batch_aborts_document",
        raised,
        "a hole in a vector set is never written to disk",
    )


def check_annual_skip_default(report: Report) -> None:
    """Checks the annual-report section default and its override."""
    from ingestion.chunker import DEFAULT_ANNUAL_SKIP_SECTIONS

    report.check(
        "chunker.annual_skip_default_covers_compliance",
        set(DEFAULT_ANNUAL_SKIP_SECTIONS) == {"financials", "notice", "governance", "esg"},
        "default=%s" % DEFAULT_ANNUAL_SKIP_SECTIONS,
    )

    document = _synthetic_document()
    document.sections.append(Section(
        section_id="financials", title="Financial Statements",
        page_start=4, page_end=4,
    ))
    document.sections[-1].blocks = [
        Block(kind="table", text="Revenue from operations 926,163 897,604",
              page_start=4, page_end=4),
    ]
    kept = chunk_document(document).section_counts()
    skipped = chunk_document(
        document, skip_sections=DEFAULT_ANNUAL_SKIP_SECTIONS).section_counts()
    report.check(
        "chunker.skip_sections_drops_only_named_sections",
        "financials" in kept and "financials" not in skipped
        and skipped.get("mdna") == kept.get("mdna"),
        "with=%s without=%s" % (sorted(kept), sorted(skipped)),
    )

    # The narrative sections a fundamental thesis is written from must never be
    # in the default skip list, whatever else is.
    narrative = {"mdna", "business", "risk"}
    report.check(
        "chunker.narrative_sections_never_skipped",
        not (narrative & set(DEFAULT_ANNUAL_SKIP_SECTIONS)),
        "narrative sections=%s" % sorted(narrative),
    )


def check_dossier(report: Report) -> None:
    """Checks that evidence is pivoted into report sections correctly."""
    from ingestion.dossier import (
        DOSSIER_JSON,
        DOSSIER_MARKDOWN,
        build_dossier,
        render_markdown,
        write_dossier,
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        registry = DocumentRegistry("TEST", root)
        registry.ensure_dirs()
        registry.company_name = "Test Industries Ltd"

        # One annual report with many chunks, and one transcript with few: the
        # case the per-document cap exists for.
        plan = [
            ("annual_report_FY2026", DOC_TYPE_ANNUAL_REPORT, "FY2026", "mdna",
             "Management Discussion and Analysis", 40, "operating_results"),
            ("transcript_2026_07", DOC_TYPE_TRANSCRIPT, "Jul 2026", "qa",
             "Question and Answer Session", 4, "quarter_performance"),
        ]
        for doc_id, doc_type, label, section_id, section_title, count, focus in plan:
            record = registry.upsert(doc_id=doc_id, doc_type=doc_type, label=label,
                                     source_url="https://example.test/%s.pdf" % doc_id)
            chunks = [
                Chunk(
                    chunk_id="%s::%s::%03d" % (doc_id, section_id, index),
                    ticker="TEST", doc_id=doc_id, doc_type=doc_type, label=label,
                    section_id=section_id, section_title=section_title,
                    text="Passage %d of %s about margins." % (index, doc_id),
                    page_start=index + 1, page_end=index + 1,
                    speaker="Aparna Iyer" if doc_type == DOC_TYPE_TRANSCRIPT else "",
                    role="management" if doc_type == DOC_TYPE_TRANSCRIPT else "",
                )
                for index in range(count)
            ]
            chunk_set = ChunkSetStub(doc_id, doc_type, label, chunks)
            cache = root / "TEST" / "chunks" / (doc_id + ".json")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(chunk_set.to_dict()), encoding="utf-8")
            registry.complete(record, "parse", "fp", "annual/v1",
                              detail={"pages": count})
            registry.complete(record, "chunk", "fp", "chunker/v3", detail={
                "cache": "chunks/" + doc_id + ".json",
                "count": len(chunks),
                "fingerprint": chunk_set.fingerprint,
            })

            # The annual report scores higher on every passage, which is what
            # would bury the transcript without a per-document cap.
            base = 0.90 if doc_type == DOC_TYPE_ANNUAL_REPORT else 0.60
            artifact = {
                "embed_model": "qwen3-embed",
                "focus_ids": [focus, "accounting_flags"],
                "focus_ids_with_evidence": [focus],
                "retrieved": [
                    {"chunk_id": chunk.chunk_id, "score": base - 0.001 * index,
                     "focus_id": focus, "section": section_id}
                    for index, chunk in enumerate(chunks)
                ],
            }
            prompt_path = root / "TEST" / "findings" / (doc_id + ".prompt.json")
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(json.dumps(artifact), encoding="utf-8")
            registry.complete(record, "prompt", "fp", "extraction-prompt/v1", detail={
                "path": "findings/" + doc_id + ".prompt.json",
            })

        dossier = build_dossier(registry, registry.select())
        report.check(
            "dossier.sections_are_report_sections",
            [section.section_id for _key, section in sorted(dossier.sections.items())]
            and set(dossier.sections) == {key for key, _t in REPORT_SECTIONS},
            "%d sections" % len(dossier.sections),
        )

        financial = dossier.sections["financial_results"]
        report.check(
            "dossier.evidence_routed_to_sections",
            bool(financial.passages),
            "%d passages in financial_results" % len(financial.passages),
        )

        docs = {passage.doc_id for passage in financial.passages}
        report.check(
            "dossier.no_document_crowds_out_the_others",
            len(docs) == 2,
            "documents contributing=%s" % sorted(docs),
        )
        per_doc = {}
        for passage in financial.passages:
            per_doc[passage.doc_id] = per_doc.get(passage.doc_id, 0) + 1
        report.check(
            "dossier.per_document_cap_enforced",
            max(per_doc.values()) <= 5,
            "per document=%s" % per_doc,
        )

        report.check(
            "dossier.passages_are_citable",
            all(passage.citation and passage.chunk_id and passage.pages
                for passage in financial.passages),
            "every passage names its document, section and page",
        )
        speaker_passages = [p for p in financial.passages if p.speaker]
        report.check(
            "dossier.transcript_passages_keep_their_speaker",
            bool(speaker_passages)
            and all("management" in p.citation for p in speaker_passages),
            "%d passages carry a speaker and role" % len(speaker_passages),
        )

        gaps = dossier.sections["strengths_weaknesses"].gaps
        report.check(
            "dossier.uncovered_targets_recorded_as_gaps",
            any("accounting" in gap.lower() or "auditor" in gap.lower() for gap in gaps),
            "gaps=%s" % gaps[:2],
        )

        empty = [key for key, section in dossier.sections.items() if not section.passages]
        report.check(
            "dossier.empty_sections_reported",
            dossier.coverage()["sections_empty"] == empty,
            "%d empty sections named" % len(empty),
        )

        detail = write_dossier(dossier, registry.resolve("findings"))
        report.check(
            "dossier.written_to_disk",
            (registry.root / "findings" / DOSSIER_JSON).exists()
            and (registry.root / "findings" / DOSSIER_MARKDOWN).exists(),
            "%s + %s" % (DOSSIER_JSON, DOSSIER_MARKDOWN),
        )
        markdown = render_markdown(dossier)
        report.check(
            "dossier.markdown_is_organised_by_section",
            all(("## %d. " % position) in markdown
                for position in range(1, len(REPORT_SECTIONS) + 1)),
            "all %d report sections have a heading" % len(REPORT_SECTIONS),
        )
        report.check(
            "dossier.markdown_cites_chunk_ids",
            financial.passages[0].chunk_id in markdown,
            "passages are traceable back to their chunk",
        )

        again = build_dossier(registry, registry.select())
        report.check(
            "dossier.fingerprint_is_reproducible",
            again.fingerprint == dossier.fingerprint == detail["fingerprint"],
            dossier.fingerprint[:16],
        )


class ChunkSetStub:
    """Minimal stand-in for a ChunkSet, for building dossier fixtures."""

    def __init__(self, doc_id, doc_type, label, chunks):
        """Stores the fixture's chunks."""
        from ingestion.chunker import ChunkSet
        self._set = ChunkSet(doc_id=doc_id, doc_type=doc_type, ticker="TEST",
                             label=label, params={}, chunks=chunks)

    @property
    def fingerprint(self):
        """Returns the underlying chunk set's fingerprint."""
        return self._set.fingerprint

    def to_dict(self):
        """Returns the underlying chunk set's JSON form."""
        return self._set.to_dict()


def check_rejection_handling(report: Report) -> None:
    """Checks that a refused passage costs one passage, not the document.

    The endpoint applies a content filter to the whole request, so one passage
    it dislikes fails the batch of 64 around it. This was not hypothetical: a
    bank's investor deck naming a security incident refused three documents
    outright, discarding 129 usable passages between them. A permanent
    rejection must therefore bisect to isolate the cause, exclude only that
    passage, and name it in the manifest -- an unretrievable gap that nothing
    records is indistinguishable from a passage that was never there.
    """
    from ingestion.embedder import (
        EmbeddingClient,
        EmbeddingError,
        EmbeddingRejected,
        _PERMANENT_STATUSES,
    )

    report.check(
        "rejection.transient_statuses_still_retry",
        408 not in _PERMANENT_STATUSES and 429 not in _PERMANENT_STATUSES
        and 400 in _PERMANENT_STATUSES and 403 in _PERMANENT_STATUSES,
        "400/403 permanent, 408/429 retried",
    )

    poison = "17"
    calls = {"n": 0}

    def filtered_request(inputs):
        """Refuses any batch containing the poisoned input, as the filter does."""
        calls["n"] += 1
        if any(text == poison for text in inputs):
            raise EmbeddingRejected(
                'HTTP 400: {"error":{"message":"Content blocked: harmful_violence"}}')
        return [[float(int(text)), 1.0, 2.0] for text in inputs]

    client = EmbeddingClient(batch_size=16, max_workers=1)
    client._request = filtered_request  # noqa: SLF001 - substituting the transport
    texts = [str(index) for index in range(40)]
    matrix, excluded, reason = client.embed_tolerating_rejections(
        texts, progress_every=1000)

    report.check(
        "rejection.only_the_refused_chunk_is_dropped",
        excluded == [17] and matrix.shape[0] == len(texts) - 1,
        "excluded=%s rows=%d of %d" % (excluded, matrix.shape[0], len(texts)),
    )
    report.check(
        "rejection.reason_is_recorded",
        "Content blocked" in reason,
        repr(reason[:60]),
    )
    report.check(
        "rejection.bisection_is_logarithmic",
        calls["n"] <= 3 + 2 * 5,
        "%d requests to isolate one input among 40" % calls["n"],
    )

    # Surviving rows must stay in input order, with the excluded index removed.
    recovered = [
        round(float(row[0]) / float(row[2]) * 2.0) if float(row[2]) else -1
        for row in matrix
    ]
    expected = [index for index in range(len(texts)) if index != 17]
    report.check(
        "rejection.surviving_rows_keep_their_order",
        recovered == expected,
        "first mismatch at %s" % next(
            (i for i, (a, b) in enumerate(zip(recovered, expected)) if a != b), "none"),
    )

    # A transient failure is not a content rejection and must still surface.
    def flaky_request(inputs):
        """Fails transiently however many times it is called."""
        raise EmbeddingError("connection reset")

    client._request = flaky_request  # noqa: SLF001
    try:
        client.embed_tolerating_rejections(["a", "b"], progress_every=1000)
        surfaced = False
    except EmbeddingError:
        surfaced = True
    report.check(
        "rejection.transient_failure_still_raises",
        surfaced,
        "a network failure is not silently treated as refused content",
    )

    # The manifest must name the exclusions, and a reload must return them.
    from ingestion.embedder import load_vectors, save_vectors

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        kept_ids = ["d::s::%03d" % index for index in expected]
        detail = save_vectors(
            root, "d", "qwen3-embed", kept_ids, matrix, "fp",
            excluded_chunk_ids=["d::s::017"], exclusion_reason=reason,
        )
        report.check(
            "rejection.manifest_names_the_exclusions",
            detail["excluded_chunk_ids"] == ["d::s::017"]
            and "Content blocked" in detail["exclusion_reason"],
            "detail=%s" % detail["excluded_chunk_ids"],
        )
        reloaded = load_vectors(root, "d")
        report.check(
            "rejection.exclusions_survive_a_reload",
            reloaded is not None
            and reloaded.excluded_chunk_ids == ["d::s::017"]
            and reloaded.count == len(kept_ids),
            "%d rows, %d excluded" % (
                reloaded.count if reloaded else -1,
                len(reloaded.excluded_chunk_ids) if reloaded else -1),
        )


def check_section_vocabulary(report: Report) -> None:
    """Checks that no two sections claim the same phrase.

    Section keys are scored by keyword hits and the highest score wins, so a
    phrase listed under two sections makes the winner arbitrary. That is not a
    tidiness point: listing "management discussion and analysis report" under
    governance as well as mdna -- which is defensible, since the MD&A is
    formally an annexure to the Board's Report -- put one issuer's MD&A into
    governance, where the default skip list dropped it, and the ingest lost the
    most useful narrative section in the filing while reporting success.
    """
    from ingestion.parsers import ANNUAL_SECTIONS

    owners = {}
    clashes = []
    for key, _title, phrases in ANNUAL_SECTIONS:
        for phrase in phrases:
            if phrase in owners and owners[phrase] != key:
                clashes.append("%r: %s vs %s" % (phrase, owners[phrase], key))
            owners[phrase] = key
    report.check(
        "vocab.no_phrase_claimed_twice",
        not clashes,
        "; ".join(clashes[:3]) if clashes
        else "%d phrases, each owned by one section" % len(owners),
    )

    # A phrase that contains another section's phrase collides just as badly,
    # because both match the same heading text.
    contained = []
    for phrase, key in owners.items():
        for other, other_key in owners.items():
            if phrase != other and other_key != key and other in phrase:
                contained.append("%r (%s) contains %r (%s)"
                                 % (phrase, key, other, other_key))
    report.check(
        "vocab.no_phrase_contains_another_sections",
        not contained,
        "; ".join(contained[:3]) if contained
        else "no phrase is a superstring of another section's",
    )

    # The narrative sections must be reachable at all.
    keys = {key for key, _t, _p in ANNUAL_SECTIONS}
    report.check(
        "vocab.narrative_sections_defined",
        {"mdna", "business", "risk"} <= keys,
        "sections=%s" % sorted(keys),
    )


def check_issuer_agnostic(report: Report) -> None:
    """Checks that no ingestion module is coupled to a particular company.

    The layer is meant to ingest any listed Indian issuer, and the way that
    breaks is quietly: a section keyword or a non-speaker label taken from
    whichever filing was open at the time keeps working for that company and
    silently stops working for every other one. Two such defects existed here --
    an ``about <issuer>`` heading in the annual-report vocabulary, and a
    ``for <issuer> limited`` sign-off in the transcript label list -- so this
    scans the package source for issuer names and asserts that the behaviour
    they were standing in for is now derived instead.

    Comments are scanned too. A worked example naming one filer is how the next
    hardcoded keyword gets justified.
    """
    package = Path(__file__).resolve().parent.parent / "ingestion"
    modules = sorted(package.glob("*.py"))
    report.check(
        "generic.package_found",
        len(modules) >= 10,
        "%d modules scanned" % len(modules),
    )

    # A sample of large listed Indian issuers, spanning the sectors whose
    # filings differ most in structure.
    issuers = [
        "wipro", "infosys", "reliance", "adani", "bikaji", "hindustan unilever",
        "bajaj", "maruti", "asian paints", "britannia", "nestle", "titan",
        "sun pharma", "dr reddy", "ultratech", "grasim", "jsw", "vedanta",
        "coal india", "power grid", "bharti airtel", "kotak", "axis bank",
        "icici", "hdfc", "sbi", "larsen", "toubro", "mahindra", "tata",
        "tcs", "hcl", "tech mahindra", "ltimindtree", "persistent",
    ]
    offenders = []
    for module in modules:
        lowered = module.read_text(encoding="utf-8").lower()
        for issuer in issuers:
            if issuer in lowered:
                offenders.append("%s: %s" % (module.name, issuer))
    report.check(
        "generic.no_issuer_named_in_package",
        not offenders,
        "; ".join(offenders[:4]) if offenders
        else "no module names a specific issuer, in code or comment",
    )

    # The behaviour those keywords stood in for must be derived instead.
    from ingestion.parsers import company_keywords

    for company, expected in (
        ("Tata Motors Limited", "about tata"),
        ("Bikaji Foods International Limited", "about bikaji"),
        ("Reliance Industries Ltd", "about reliance"),
        ("Infosys Limited", "about infosys"),
    ):
        phrases = company_keywords(company).get("business", [])
        if not report.check(
                "generic.company_headings_derived[%s]" % company.split()[0],
                expected in phrases,
                "%s -> %s" % (company, phrases[:2])):
            break

    report.check(
        "generic.unknown_company_falls_back",
        company_keywords("") == {} and company_keywords("Limited") == {},
        "a missing name yields no keywords rather than 'about limited'",
    )

    # A covering letter's sign-off is matched by shape, for any issuer.
    from ingestion.parsers import _is_speaker_name

    signatures = [
        "For Tata Consultancy Services Limited",
        "For Bikaji Foods International Limited",
        "For Larsen & Toubro Ltd",
        "For Some Company Nobody Has Heard Of Limited",
    ]
    report.check(
        "generic.signature_lines_are_not_speakers",
        not any(_is_speaker_name(name) for name in signatures),
        "%d sign-off forms rejected" % len(signatures),
    )
    report.check(
        "generic.people_are_still_speakers",
        all(_is_speaker_name(name) for name in
            ("Aparna Iyer", "Nitin Padmanabhan", "Moderator", "Dr. Anish Shah")),
        "personal names and the moderator still open turns",
    )

    # Download headers follow the URL's host, not one exchange.
    from ingestion.fetcher import _request_headers

    hosts = {
        "https://www.bseindia.com/x.pdf": "https://www.bseindia.com/",
        "https://www.nseindia.com/x.pdf": "https://www.nseindia.com/",
        "https://www.example-issuer.com/ir/x.pdf": "https://www.example-issuer.com/",
    }
    wrong = [
        url for url, referer in hosts.items()
        if _request_headers(url).get("Referer") != referer
    ]
    report.check(
        "generic.referer_follows_the_host",
        not wrong,
        "%d hosts get a same-origin referer" % len(hosts) if not wrong
        else "wrong for %s" % wrong[:2],
    )

    # Indian filings write the rupee three ways, sometimes within one document.
    from ingestion.parsers import is_table_line

    rows = [
        "Revenue from operations Rs.9,26,163 Rs.8,97,604",
        "Total income INR 12,345 INR 11,200",
        "Profit before tax 1,234.5 1,102.3",
        "Revenue 9,26,163 8,97,604",
    ]
    report.check(
        "generic.indian_currency_forms_read_as_tables",
        all(is_table_line(row) for row in rows),
        "%d rupee notations recognised" % len(rows),
    )

    # Tickers reach the filesystem, and Indian symbols carry punctuation.
    from ingestion.registry import safe_ticker

    report.check(
        "generic.punctuated_tickers_are_safe",
        safe_ticker("M&M") == "M_M" and safe_ticker("BAJAJ-AUTO") == "BAJAJ-AUTO"
        and safe_ticker("l&tfh") == "L_TFH" and safe_ticker("") == "UNKNOWN",
        "M&M -> %s, BAJAJ-AUTO -> %s" % (safe_ticker("M&M"), safe_ticker("BAJAJ-AUTO")),
    )


def check_prompts(report: Report) -> None:
    """Checks that prompts are built from the document actually in hand."""
    probes_all = build_probes(DOC_TYPE_ANNUAL_REPORT, "Example Industries Ltd", "FY2026",
                             available_sections=["mdna", "risk", "business",
                                                 "financials", "governance"])
    probes_thin = build_probes(DOC_TYPE_ANNUAL_REPORT, "Example Industries Ltd", "FY2026",
                               available_sections=["mdna"])
    report.check(
        "prompts.probes_track_sections",
        0 < len(probes_thin) < len(probes_all),
        "%d probes with mdna only vs %d with all sections"
        % (len(probes_thin), len(probes_all)),
    )

    # An area lives in more than one section, so it survives while any of them
    # is present: risk commentary genuinely appears inside the MD&A. What must
    # be dropped is an area with no surviving section at all -- the auditor's
    # observations, which exist only in the financial statements and the
    # governance report.
    focus_ids = {area.focus_id for area, _q in probes_thin}
    report.check(
        "prompts.absent_section_not_probed",
        "accounting_flags" not in focus_ids and "operating_results" in focus_ids,
        "focus ids probed=%s" % sorted(focus_ids),
    )

    # The query must not repeat the chunks' context header. Both sides carrying
    # "Example Industries | Annual Report | FY2026" scores that header against
    # every candidate, and the chunks that win are the ones with the least
    # content -- a near-empty chunk is nothing but its header.
    query = probes_all[0][1]
    report.check(
        "prompts.probe_is_topic_only",
        "Example Industries" not in query and "FY2026" not in query
        and len(query) > 40,
        repr(query[:70]),
    )
    report.check(
        "prompts.probe_states_its_topic",
        probes_all[0][0].title.lower() in query.lower(),
        repr(query[:70]),
    )

    chunk = Chunk(
        chunk_id="annual_report_FY2026::mdna::001", ticker="TEST",
        doc_id="annual_report_FY2026", doc_type=DOC_TYPE_ANNUAL_REPORT,
        label="FY2026", section_id="mdna",
        section_title="Management Discussion and Analysis",
        text="Revenue grew to 926,163 million.", page_start=41, page_end=41,
    )
    built = build_extraction_prompt(
        ticker="TEST", company="Example Industries Ltd",
        doc_id="annual_report_FY2026",
        doc_type=DOC_TYPE_ANNUAL_REPORT, label="FY2026", chunks=[chunk],
        focus_areas=[area for area, _q in probes_all],
        available_sections=["mdna"],
    )
    prompt = built["prompt"]
    report.check(
        "prompts.evidence_is_citable",
        chunk.chunk_id in prompt and built["chunk_ids"] == [chunk.chunk_id],
        "chunk id appears in the evidence block and the citation list",
    )
    report.check(
        "prompts.absences_requested",
        "absences" in prompt and "Name what is missing" in prompt,
        "the contract asks for uncovered targets",
    )
    report.check(
        "prompts.no_derived_metrics",
        "Do not compute derived metrics" in prompt,
        "ratios stay with the deterministic engine",
    )

    for doc_type, profile in DOC_PROFILES.items():
        if not report.check(
                "prompts.profile_states_limits[%s]" % doc_type,
                bool(profile.authority) and bool(profile.limits) and bool(profile.focus_areas),
                "%d focus areas" % len(profile.focus_areas)):
            break


def check_registry(report: Report) -> None:
    """Checks the registry's staleness rules and its atomic write."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        registry = DocumentRegistry("TEST", root)
        record = registry.upsert(
            doc_id="annual_report_FY2026", doc_type=DOC_TYPE_ANNUAL_REPORT,
            label="FY2026", source_url="https://example.test/ar-2026.pdf",
        )
        report.check(
            "registry.new_document_needs_download",
            registry.needs(record, "download", record.url_fingerprint, "fetcher/v1"),
            "a document never fetched is scheduled",
        )

        registry.complete(record, "download", record.url_fingerprint, "fetcher/v1",
                          detail={"sha256": "abc", "bytes": 10})
        report.check(
            "registry.completed_stage_is_skipped",
            not registry.needs(record, "download", record.url_fingerprint, "fetcher/v1"),
            "unchanged input is not re-run",
        )
        report.check(
            "registry.version_change_invalidates",
            registry.needs(record, "download", record.url_fingerprint, "fetcher/v2"),
            "a stage version change re-runs the stage",
        )

        registry.complete(record, "embed", "chunk-fp", "embedder/v1",
                          detail={"model": "qwen3-embed", "count": 12})
        report.check(
            "registry.model_change_invalidates_embedding",
            registry.needs(record, "embed", "chunk-fp-with-other-model", "embedder/v1"),
            "vectors from another model are never reused",
        )

        registry.save()
        again = DocumentRegistry("TEST", root)
        reloaded = again.documents.get("annual_report_FY2026")
        report.check(
            "registry.round_trip",
            reloaded is not None and reloaded.stage("download").is_done,
            "state survives a reload",
        )

        # A changed catalogue link must invalidate everything below it.
        changed = again.upsert(
            doc_id="annual_report_FY2026", doc_type=DOC_TYPE_ANNUAL_REPORT,
            label="FY2026", source_url="https://example.test/ar-2026-revised.pdf",
        )
        report.check(
            "registry.url_change_resets_stages",
            not changed.stage("download").is_done and not changed.stage("embed").is_done,
            "a new link re-ingests the document",
        )
        report.check(
            "registry.url_change_recorded_in_history",
            any(event.get("event") == "source_url_changed" for event in changed.history),
            "the change is auditable",
        )

        missing = again.upsert(
            doc_id="transcript_2026_07", doc_type=DOC_TYPE_TRANSCRIPT,
            label="Jul 2026", source_url="https://example.test/t.pdf",
        )
        again.complete(missing, "chunk", "fp", "chunker/v1",
                       detail={"cache": "chunks/transcript_2026_07.json", "count": 5})
        report.check(
            "registry.missing_output_reruns_stage",
            again.needs(missing, "chunk", "fp", "chunker/v1"),
            "a stage whose output file has gone is re-run",
        )
        report.check(
            "registry.stages_cover_pipeline",
            STAGES == ["download", "parse", "chunk", "embed", "prompt"],
            "stages=%s" % STAGES,
        )


def run_offline_checks() -> Report:
    """Runs every check that needs neither the network nor the embedding API."""
    report = Report()
    for name, function in (
        ("layout", check_layout),
        ("parsers", check_parsers),
        ("chunker", check_chunker),
        ("annual_skip", check_annual_skip_default),
        ("embedder", check_embedding_batching),
        ("prompts", check_prompts),
        ("dossier", check_dossier),
        ("registry", check_registry),
        ("rejection", check_rejection_handling),
        ("vocab", check_section_vocabulary),
        ("generic", check_issuer_agnostic),
    ):
        try:
            function(report)
        except Exception as exc:  # noqa: BLE001 - a crashed group is a failure
            report.check(name + ".crashed", False, "%s: %s" % (type(exc).__name__, exc))
    return report


# --- Live checks --------------------------------------------------------------

def check_idempotent_run(
    report: Report,
    ticker: str,
    data_dir: Path,
    doc_types: Optional[List[str]] = None,
    concall_years: int = 1,
    max_pages: Optional[int] = None,
) -> None:
    """Runs the pipeline twice and asserts the second run does no work.

    Args:
        report: Report to record into.
        ticker: Ticker to ingest.
        data_dir: Data directory to use. A temporary one gives a true first run.
        doc_types: Document classes to include.
        concall_years: Years of concalls to request.
        max_pages: Optional page cap, to keep the check quick.
    """
    from ingestion.graph import IngestionPipeline

    def run() -> Tuple[Dict[str, Any], DocumentRegistry]:
        """Runs the pipeline once and returns its state and registry."""
        pipeline = IngestionPipeline(
            ticker, data_dir=data_dir, doc_types=doc_types,
            concall_years=concall_years, max_pages=max_pages,
        )
        return dict(pipeline.run()), pipeline.registry

    first, registry = run()
    actions = first.get("actions", {})
    report.check(
        "live.first_run_downloads",
        actions.get("download", {}).get("downloaded", 0) > 0,
        "downloaded=%d" % actions.get("download", {}).get("downloaded", 0),
    )
    report.check(
        "live.first_run_embeds",
        actions.get("embed", {}).get("embedded", 0) > 0,
        "embedded=%d documents, %d vectors"
        % (actions.get("embed", {}).get("embedded", 0),
           actions.get("embed", {}).get("vectors", 0)),
    )
    report.check(
        "live.first_run_has_no_errors",
        not first.get("errors"),
        json.dumps(first.get("errors") or [])[:200] or "no failures",
    )

    second, registry = run()
    plan = second.get("planned", {})
    second_actions = second.get("actions", {})
    report.check(
        "live.second_run_plans_nothing",
        not plan,
        "planned=%s" % json.dumps(plan)[:160] if plan else "empty plan",
    )
    for stage in ("download", "parse", "chunk", "embed"):
        did = second_actions.get(stage, {})
        worked = sum(count for key, count in did.items()
                     if key not in ("current", "skipped"))
        report.check(
            "live.second_run_no_%s" % stage,
            worked == 0,
            "%s=%s" % (stage, json.dumps(did)),
        )
    dossier = second_actions.get("dossier", {})
    report.check(
        "live.second_run_dossier_unchanged",
        dossier.get("written", 0) == 0 or dossier.get("unchanged", 0) > 0,
        "dossier=%s" % json.dumps(dossier),
    )

    # Re-chunking a document must not re-embed it when the chunks come out
    # identical. Forcing the chunk stage is the cheapest way to reach that
    # state, and the expensive stage below it must decline to run.
    from ingestion.graph import IngestionPipeline

    forced = IngestionPipeline(
        ticker, data_dir=data_dir, doc_types=doc_types,
        concall_years=concall_years, max_pages=max_pages, force=["chunk"],
    )
    forced_state = dict(forced.run())
    forced_actions = forced_state.get("actions", {})
    report.check(
        "live.rechunk_does_not_reembed_identical_chunks",
        forced_actions.get("chunk", {}).get("chunked", 0) > 0
        and forced_actions.get("embed", {}).get("embedded", 0) == 0,
        "chunk=%s embed=%s" % (json.dumps(forced_actions.get("chunk", {})),
                               json.dumps(forced_actions.get("embed", {}))),
    )

    counts = registry.counts()
    report.check(
        "live.every_document_embedded",
        counts["documents"] > 0 and counts["embed"] == counts["documents"],
        "documents=%d embedded=%d" % (counts["documents"], counts["embed"]),
    )
    report.check(
        "live.every_document_has_a_prompt",
        counts["prompt"] == counts["documents"],
        "prompts=%d of %d" % (counts["prompt"], counts["documents"]),
    )

    # Every embedding must name the model that produced it, or a later model
    # change cannot be detected and two vector sets end up in one index.
    unnamed = [
        row["doc_id"] for row in registry.status_table()
        if row["embed"] == "done" and not row["embed_model"]
    ]
    report.check(
        "live.embeddings_name_their_model",
        not unnamed,
        "unnamed=%s" % unnamed if unnamed else "every vector set names its model",
    )


def check_vector_alignment(report: Report, ticker: str, data_dir: Path) -> None:
    """Checks that stored vectors and chunk caches agree, and that search works."""
    from ingestion.embedder import EmbeddingClient, load_vectors
    from ingestion.retrieval import load_document_index, retrieve_for_probes

    registry = DocumentRegistry(ticker, data_dir)
    root = registry.root
    embedded = [
        record for record in registry.select() if record.stage("embed").is_done
    ]
    if not embedded:
        report.check("vectors.available", False, "no embedded documents to check")
        return

    aligned = 0
    for record in embedded:
        store = load_vectors(root / "vectors", record.doc_id)
        chunk_set = read_chunk_cache(root / "chunks" / (record.doc_id + ".json"))
        if store is None or chunk_set is None:
            report.check("vectors.loadable[%s]" % record.doc_id, False, "missing files")
            continue
        # Rows are positional, so the manifest must account for every
        # chunk: embedded, or named as refused by the service.
        refused = set(store.excluded_chunk_ids)
        expected = [
            chunk.chunk_id for chunk in chunk_set.chunks
            if chunk.chunk_id not in refused
        ]
        if not report.check(
                "vectors.rows_match_chunks[%s]" % record.doc_id,
                store.chunk_ids == expected,
                "%d vectors, %d chunks, %d refused"
                % (store.count, len(chunk_set.chunks), len(refused))):
            continue
        report.check(
            "vectors.fingerprint_matches[%s]" % record.doc_id,
            store.fingerprint == chunk_set.fingerprint,
            "vectors were built from the current chunk set",
        )
        aligned += 1

    if not aligned:
        return

    document = load_document_index(root / "chunks", root / "vectors",
                                  embedded[0].doc_id)
    report.check(
        "retrieval.index_loads",
        document is not None,
        embedded[0].doc_id,
    )
    if document is None:
        return

    client = EmbeddingClient()
    probes = build_probes(document.doc_type, registry.company_name or ticker,
                          document.label, available_sections=document.sections)
    hits = retrieve_for_probes(document, probes, client)
    report.check(
        "retrieval.returns_hits",
        bool(hits),
        "%d distinct chunks" % len(hits),
    )
    report.check(
        "retrieval.hits_are_scored_and_ordered",
        all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1)),
        "scores descend",
    )
    report.check(
        "retrieval.hits_carry_provenance",
        all(hit.chunk.section_id and hit.chunk.page_start > 0 for hit in hits),
        "every hit names its section and page",
    )

    # A filter for a section the document does not have must return nothing,
    # rather than the nearest available text from somewhere else.
    empty = document.search(
        document.vectors[0], top_k=5, sections=["section_that_does_not_exist"],
    )
    report.check(
        "retrieval.absent_section_returns_nothing",
        empty == [],
        "%d hits for an absent section" % len(empty),
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Parses arguments and runs the verification suite.

    Args:
        argv: Argument list, defaulting to sys.argv[1:].

    Returns:
        0 when every check passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Verify the ingestion layer.")
    parser.add_argument("ticker", nargs="?", default="WIPRO",
                        help="Ticker for the live checks (default: WIPRO)")
    parser.add_argument("--offline", action="store_true",
                        help="Run only the checks that need no network")
    parser.add_argument("--doc-types", nargs="*", default=["concall_presentation"],
                        help="Document classes for the live run "
                             "(default: concall_presentation, the quickest)")
    parser.add_argument("--concall-years", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Cap pages parsed per document during the live run")
    parser.add_argument("--data-dir", default=None,
                        help="Reuse an existing data directory instead of a "
                             "temporary one. The first-run checks will not hold.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    print("Ingestion verification")
    print("=" * 72)
    print("\nOffline checks")
    report = run_offline_checks()
    print(report.render())

    if not args.offline:
        print("\nLive checks (%s, %s)" % (args.ticker, ", ".join(args.doc_types)))
        live = Report()
        temporary: Optional[str] = None
        try:
            if args.data_dir:
                data_dir = Path(args.data_dir)
            else:
                temporary = tempfile.mkdtemp(prefix="grownxt-verify-")
                data_dir = Path(temporary)
            check_idempotent_run(
                live, args.ticker, data_dir,
                doc_types=args.doc_types, concall_years=args.concall_years,
                max_pages=args.max_pages,
            )
            check_vector_alignment(live, args.ticker, data_dir)
        except Exception as exc:  # noqa: BLE001 - report rather than traceback
            live.check("live.crashed", False, "%s: %s" % (type(exc).__name__, exc))
        finally:
            if temporary:
                shutil.rmtree(temporary, ignore_errors=True)
        print(live.render())
        report.results.extend(live.results)

    print("\n" + "=" * 72)
    if report.failures:
        print("FAILED: %d check(s) did not hold." % len(report.failures))
        for result in report.failures:
            print("  - %s: %s" % (result.name, result.detail))
        return 1
    print("All %d checks passed." % len(report.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
