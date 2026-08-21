"""Evidence dossier: ingested filings reorganised into report sections.

The ingestion layer stores evidence the way the *documents* are organised -- one
chunk set and one prompt per filing. A report author needs the opposite view:
everything the filings say about capital allocation, gathered in one place,
whichever document said it. This module pivots the one into the other.

The dossier is the hand-off point between this layer and report generation. Its
sections are the report generator's own section keys, so a passage arrives where
it is needed without translation, and each passage keeps the citation it was
retrieved with -- document, section, page range, and for a call, the speaker and
their role.

Three properties make it usable as a research input rather than a text dump:

* **It is derived, not re-retrieved.** Everything comes from the prompt
  artifacts and chunk caches already on disk, so building it costs no embedding
  calls and produces the same dossier from the same ingest every time.
* **No single document may crowd out the others.** An annual report has an order
  of magnitude more chunks than an earnings call, so on score alone it wins
  every section -- burying the more recent and more specific commentary. A cap
  per document per section keeps the newest call's evidence present. That is a
  property of the document classes, not of any one filer.
* **Gaps are stated.** A section with no evidence is recorded as empty, and a
  target that retrieved nothing is named. A report author needs to know that the
  filings are silent on a topic, which is not the same as the ingest having
  skipped it.

Google Python Style Guide Compliant.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ingestion.chunker import DOC_TYPE_LABELS, Chunk, read_chunk_cache
from ingestion.prompts import DOC_PROFILES, REPORT_SECTIONS, PROMPT_VERSION
from ingestion.registry import DocumentRecord, DocumentRegistry, sha256_text, utc_now

logger = logging.getLogger(__name__)

DOSSIER_VERSION: str = "dossier/v1"

DOSSIER_JSON: str = "evidence_dossier.json"
DOSSIER_MARKDOWN: str = "evidence_dossier.md"

# Passages kept per report section. Enough to write a section from, few enough
# that the section is a reading list rather than a corpus.
MAX_PER_SECTION: int = 12

# Passages one document may contribute to one section. This is what stops a
# 561-page annual report from filling every section on its own.
MAX_PER_DOCUMENT: int = 5

# Document classes in the order a research author would weigh them for recency.
DOC_TYPE_ORDER: Dict[str, int] = {
    "concall_transcript": 0,
    "concall_presentation": 1,
    "annual_report": 2,
}

REPORT_SECTION_TITLES: Dict[str, str] = dict(REPORT_SECTIONS)


@dataclass
class Passage:
    """One cited passage placed in a report section.

    Attributes:
        chunk_id: Identifier of the source chunk.
        doc_id: Source document.
        doc_type: Source document class.
        label: Period label of the source document.
        source_section: Section of the source document.
        source_section_title: Display title of that section.
        pages: Printed page range, e.g. ``pp41-42``.
        focus_id: Focus area whose probe retrieved the passage.
        score: Similarity score it was retrieved at.
        text: The passage itself.
        speaker: Speaker, for a transcript turn.
        role: Speaker's role, for a transcript turn.
        title: Slide title, for a presentation passage.
    """

    chunk_id: str
    doc_id: str
    doc_type: str
    label: str
    source_section: str
    source_section_title: str
    pages: str
    focus_id: str
    score: float
    text: str
    speaker: str = ""
    role: str = ""
    title: str = ""

    @property
    def citation(self) -> str:
        """Returns a one-line citation for this passage."""
        parts = [
            DOC_TYPE_LABELS.get(self.doc_type, self.doc_type),
            self.label,
            self.source_section_title,
            self.pages,
        ]
        if self.speaker:
            parts.append("%s (%s)" % (self.speaker, self.role or "speaker"))
        elif self.title:
            parts.append(self.title)
        return " · ".join(part for part in parts if part)

    def to_dict(self) -> Dict[str, Any]:
        """Returns the JSON form of the passage."""
        out: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "label": self.label,
            "source_section": self.source_section,
            "source_section_title": self.source_section_title,
            "pages": self.pages,
            "focus_id": self.focus_id,
            "score": round(self.score, 4),
            "citation": self.citation,
            "text": self.text,
        }
        for name in ("speaker", "role", "title"):
            value = getattr(self, name)
            if value:
                out[name] = value
        return out


@dataclass
class DossierSection:
    """One report section's evidence.

    Attributes:
        section_id: Report section key.
        title: Display title.
        passages: Cited passages, strongest first.
        gaps: Targets that retrieved nothing for this section.
    """

    section_id: str
    title: str
    passages: List[Passage] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)

    def sources(self) -> List[str]:
        """Returns the distinct documents contributing to this section."""
        seen: List[str] = []
        for passage in self.passages:
            name = "%s %s" % (
                DOC_TYPE_LABELS.get(passage.doc_type, passage.doc_type), passage.label)
            if name not in seen:
                seen.append(name)
        return seen

    def to_dict(self) -> Dict[str, Any]:
        """Returns the JSON form of the section."""
        return {
            "section_id": self.section_id,
            "title": self.title,
            "n_passages": len(self.passages),
            "sources": self.sources(),
            "gaps": self.gaps,
            "passages": [passage.to_dict() for passage in self.passages],
        }


@dataclass
class Dossier:
    """A ticker's ingested evidence, organised by report section.

    Attributes:
        ticker: Owning ticker.
        company_name: Company name as catalogued.
        embed_model: Model whose vectors selected the evidence.
        sources: One entry per contributing document.
        sections: Report section key to its evidence.
    """

    ticker: str
    company_name: str
    embed_model: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    sections: Dict[str, DossierSection] = field(default_factory=dict)

    @property
    def n_passages(self) -> int:
        """Total passages across every section."""
        return sum(len(section.passages) for section in self.sections.values())

    def coverage(self) -> Dict[str, Any]:
        """Returns how much of the report structure the evidence reaches."""
        populated = [key for key, section in self.sections.items() if section.passages]
        empty = [key for key, section in self.sections.items() if not section.passages]
        return {
            "sections_total": len(self.sections),
            "sections_with_evidence": len(populated),
            "sections_empty": empty,
            "passages": self.n_passages,
            "distinct_chunks": len({
                passage.chunk_id
                for section in self.sections.values()
                for passage in section.passages
            }),
        }

    @property
    def fingerprint(self) -> str:
        """Identity of the dossier's inputs, for staleness checks."""
        payload = "|".join(sorted(
            "%s:%s" % (source.get("doc_id"), source.get("chunk_fingerprint"))
            for source in self.sources
        ))
        return sha256_text(DOSSIER_VERSION + "|" + self.embed_model + "|" + payload)

    def to_dict(self) -> Dict[str, Any]:
        """Returns the JSON form written to disk."""
        return {
            "dossier_version": DOSSIER_VERSION,
            "prompt_version": PROMPT_VERSION,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "embed_model": self.embed_model,
            "built_at": utc_now(),
            "fingerprint": self.fingerprint,
            "coverage": self.coverage(),
            "sources": self.sources,
            "sections": [
                self.sections[key].to_dict()
                for key, _title in REPORT_SECTIONS
                if key in self.sections
            ],
        }


def _page_range(chunk: Chunk) -> str:
    """Returns a printed page reference for a chunk."""
    if chunk.page_start and chunk.page_start == chunk.page_end:
        return "p%d" % chunk.page_start
    if chunk.page_start and chunk.page_end:
        return "pp%d-%d" % (chunk.page_start, chunk.page_end)
    return ""


def _focus_routing(doc_type: str) -> Dict[str, Tuple[str, List[str]]]:
    """Returns focus_id -> (title, report sections) for one document class."""
    profile = DOC_PROFILES.get(doc_type)
    if profile is None:
        return {}
    return {
        area.focus_id: (area.title, list(area.report_sections))
        for area in profile.focus_areas
    }


def _load_prompt_artifact(path: Path) -> Optional[Dict[str, Any]]:
    """Reads a built prompt artifact, or None when it is unusable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Prompt artifact unreadable at %s: %s", path, exc)
        return None


def build_dossier(
    registry: DocumentRegistry,
    records: Sequence[DocumentRecord],
    max_per_section: int = MAX_PER_SECTION,
    max_per_document: int = MAX_PER_DOCUMENT,
) -> Dossier:
    """Builds a ticker's evidence dossier from artifacts already on disk.

    Args:
        registry: The ticker's registry.
        records: Documents to draw evidence from.
        max_per_section: Passages kept per report section.
        max_per_document: Passages one document may contribute per section.

    Returns:
        The dossier. Documents whose prompt or chunk artifacts are missing are
        skipped with a warning rather than failing the build, so a partially
        ingested ticker still yields a usable dossier.
    """
    dossier = Dossier(
        ticker=registry.ticker,
        company_name=registry.company_name or registry.ticker,
        embed_model="",
    )
    for key, title in REPORT_SECTIONS:
        dossier.sections[key] = DossierSection(section_id=key, title=title)

    # Candidates per report section, before the caps are applied.
    candidates: Dict[str, List[Passage]] = {key: [] for key, _ in REPORT_SECTIONS}
    ordered = sorted(
        records,
        key=lambda record: (DOC_TYPE_ORDER.get(record.doc_type, 9), record.doc_id),
        reverse=False,
    )

    for record in ordered:
        prompt_state = record.stage("prompt")
        chunk_state = record.stage("chunk")
        if not prompt_state.is_done or not chunk_state.is_done:
            continue

        artifact = _load_prompt_artifact(
            registry.resolve(str(prompt_state.detail.get("path", ""))))
        chunk_set = read_chunk_cache(
            registry.resolve(str(chunk_state.detail.get("cache", ""))))
        if artifact is None or chunk_set is None:
            logger.warning("[%s] skipped in dossier: artifacts missing.", record.doc_id)
            continue

        by_id = {chunk.chunk_id: chunk for chunk in chunk_set.chunks}
        routing = _focus_routing(record.doc_type)
        dossier.embed_model = dossier.embed_model or str(artifact.get("embed_model", ""))
        dossier.sources.append({
            "doc_id": record.doc_id,
            "doc_type": record.doc_type,
            "label": record.label,
            "source_url": record.source_url,
            "pages": int(record.stage("parse").detail.get("pages", 0) or 0),
            "chunks": len(chunk_set.chunks),
            "chunk_fingerprint": chunk_set.fingerprint,
            "sections_present": sorted({c.section_id for c in chunk_set.chunks}),
        })

        for hit in artifact.get("retrieved") or []:
            chunk = by_id.get(str(hit.get("chunk_id", "")))
            if chunk is None:
                continue
            focus_id = str(hit.get("focus_id", ""))
            focus_title, report_sections = routing.get(focus_id, ("", []))
            passage = Passage(
                chunk_id=chunk.chunk_id,
                doc_id=record.doc_id,
                doc_type=record.doc_type,
                label=record.label,
                source_section=chunk.section_id,
                source_section_title=chunk.section_title,
                pages=_page_range(chunk),
                focus_id=focus_id,
                score=float(hit.get("score", 0.0)),
                text=chunk.text,
                speaker=chunk.speaker,
                role=chunk.role,
                title=chunk.title,
            )
            for section_id in report_sections:
                if section_id in candidates:
                    candidates[section_id].append(passage)

        # A target that retrieved nothing is a gap in every section it feeds.
        with_evidence = set(artifact.get("focus_ids_with_evidence") or [])
        for focus_id in artifact.get("focus_ids") or []:
            if focus_id in with_evidence:
                continue
            focus_title, report_sections = routing.get(str(focus_id), ("", []))
            for section_id in report_sections:
                if section_id not in dossier.sections:
                    continue
                gap = "%s: no evidence in %s %s" % (
                    focus_title or focus_id,
                    DOC_TYPE_LABELS.get(record.doc_type, record.doc_type),
                    record.label,
                )
                dossier.sections[section_id].gaps.append(gap)

    for section_id, section in dossier.sections.items():
        pool = sorted(candidates[section_id], key=lambda p: p.score, reverse=True)
        per_document: Dict[str, int] = {}
        seen: set = set()
        for passage in pool:
            if len(section.passages) >= max_per_section:
                break
            if passage.chunk_id in seen:
                continue
            used = per_document.get(passage.doc_id, 0)
            if used >= max_per_document:
                continue
            per_document[passage.doc_id] = used + 1
            seen.add(passage.chunk_id)
            section.passages.append(passage)

    logger.info("[%s] dossier: %d passages across %d of %d report sections",
                dossier.ticker, dossier.n_passages,
                dossier.coverage()["sections_with_evidence"], len(dossier.sections))
    return dossier


def render_markdown(dossier: Dossier) -> str:
    """Renders the dossier as a readable, citable Markdown document.

    The JSON form is what the report generator consumes; this is what a person
    reads to check that the evidence behind a section is the evidence they would
    have chosen.

    Args:
        dossier: The dossier to render.

    Returns:
        Markdown text.
    """
    coverage = dossier.coverage()
    lines: List[str] = [
        "# %s (%s) — Evidence Dossier" % (dossier.company_name, dossier.ticker),
        "",
        "Built %s · %d passages from %d filings · embedded with `%s`"
        % (utc_now(), dossier.n_passages, len(dossier.sources),
           dossier.embed_model or "unknown"),
        "",
        "Evidence for %d of %d report sections."
        % (coverage["sections_with_evidence"], coverage["sections_total"]),
        "",
        "## Sources",
        "",
        "| Document | Class | Period | Pages | Chunks | Sections present |",
        "| :--- | :--- | :--- | ---: | ---: | :--- |",
    ]
    for source in dossier.sources:
        lines.append("| `%s` | %s | %s | %s | %d | %s |" % (
            source["doc_id"],
            DOC_TYPE_LABELS.get(source["doc_type"], source["doc_type"]),
            source["label"],
            source["pages"] or "-",
            source["chunks"],
            ", ".join(source["sections_present"]),
        ))
    lines.append("")

    for position, (key, _title) in enumerate(REPORT_SECTIONS, start=1):
        section = dossier.sections.get(key)
        if section is None:
            continue
        lines.append("## %d. %s" % (position, section.title))
        lines.append("")
        lines.append("`%s`" % key)
        lines.append("")
        if not section.passages:
            lines.append("_No evidence retrieved for this section._")
            lines.append("")
        else:
            lines.append("%d passages · %s" % (
                len(section.passages), ", ".join(section.sources())))
            lines.append("")
            for rank, passage in enumerate(section.passages, start=1):
                lines.append("**%d. %s**  " % (rank, passage.citation))
                lines.append("_%s · similarity %.3f_" % (
                    passage.focus_id or "unattributed", passage.score))
                lines.append("")
                for line in passage.text.splitlines():
                    lines.append("> " + line if line.strip() else ">")
                lines.append("")
                lines.append("`%s`" % passage.chunk_id)
                lines.append("")
        if section.gaps:
            lines.append("<details><summary>Gaps (%d)</summary>" % len(section.gaps))
            lines.append("")
            for gap in sorted(set(section.gaps)):
                lines.append("- %s" % gap)
            lines.append("")
            lines.append("</details>")
            lines.append("")
    return "\n".join(lines)


def write_dossier(dossier: Dossier, directory: Path) -> Dict[str, Any]:
    """Writes the dossier's JSON and Markdown forms.

    Args:
        dossier: The dossier to write.
        directory: Target directory, normally ``data/<TICKER>/findings``.

    Returns:
        Detail for the registry: relative paths, counts, and the fingerprint.
    """
    directory.mkdir(parents=True, exist_ok=True)
    payload = dossier.to_dict()
    (directory / DOSSIER_JSON).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / DOSSIER_MARKDOWN).write_text(
        render_markdown(dossier), encoding="utf-8")

    coverage = payload["coverage"]
    return {
        "json_path": "findings/" + DOSSIER_JSON,
        "markdown_path": "findings/" + DOSSIER_MARKDOWN,
        "fingerprint": dossier.fingerprint,
        "passages": coverage["passages"],
        "distinct_chunks": coverage["distinct_chunks"],
        "sections_with_evidence": coverage["sections_with_evidence"],
        "sections_empty": coverage["sections_empty"],
        "sources": len(dossier.sources),
        "embed_model": dossier.embed_model,
    }
