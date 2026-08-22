"""Sell-Side Institutional Research Synthesizer for GrowNXT RAG.

Generates grounded, citable qualitative findings (Motilal Oswal, Kotak Equities style)
using the hosted LLM endpoint (https://grownxt-llm.vercel.app).

Google Python Style Guide Compliant.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import datetime as _datetime
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from core.config import safe_ticker
from core.llm_config import LLMError, generate_llm_response
from ingestion.rag.probes import PILLAR_TITLES, ThematicProbe
from ingestion.rag.retriever import EvidenceChunk

logger = logging.getLogger(__name__)


def utc_now() -> str:
    """Returns current UTC ISO-8601 string."""
    return _datetime.datetime.now(_datetime.timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ThematicFinding:
    """Structured institutional research finding for one equity research pillar."""

    pillar: str
    title: str
    bullet_points: list[str] = field(default_factory=list)
    takeaway: str = ""
    citations: list[str] = field(default_factory=list)
    raw_synthesis: str = ""
    chunks_used: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThematicFinding":
        return cls(
            pillar=str(data.get("pillar", "")),
            title=str(data.get("title", "")),
            bullet_points=list(data.get("bullet_points") or []),
            takeaway=str(data.get("takeaway", "")),
            citations=list(data.get("citations") or []),
            raw_synthesis=str(data.get("raw_synthesis", "")),
            chunks_used=list(data.get("chunks_used") or []),
        )

    def to_markdown(self) -> str:
        """Formats the finding as standard sell-side equity research markdown."""
        lines = [f"### {self.title}\n"]
        for pt in self.bullet_points:
            lines.append(f"- {pt}")
        if self.takeaway:
            lines.append(f"\n💡 **Analyst Takeaway**: {self.takeaway}")
        return "\n".join(lines)


@dataclass
class ResearchDossier:
    """Complete qualitative research dossier for one company."""

    ticker: str
    company_name: str
    generated_at: str
    pillars: dict[str, ThematicFinding] = field(default_factory=dict)
    available_doc_types: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "generated_at": self.generated_at,
            "pillars": {k: v.to_dict() for k, v in self.pillars.items()},
            "available_doc_types": list(self.available_doc_types),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchDossier":
        raw_pillars = data.get("pillars") or {}
        pillars = {
            k: ThematicFinding.from_dict(v)
            for k, v in raw_pillars.items()
            if isinstance(v, dict)
        }
        return cls(
            ticker=str(data.get("ticker", "")),
            company_name=str(data.get("company_name", "")),
            generated_at=str(data.get("generated_at", utc_now())),
            pillars=pillars,
            available_doc_types=list(data.get("available_doc_types") or []),
            summary=str(data.get("summary", "")),
        )

    def to_markdown(self) -> str:
        """Formats the full research dossier into institutional markdown."""
        parts = [
            f"# Qualitative Equity Research Dossier: {self.ticker}",
            f"> **Generated via Institutional RAG Pipeline** | As of: {self.generated_at}\n",
        ]
        if self.summary:
            parts.append(f"## Executive Thesis\n{self.summary}\n")

        for finding in self.pillars.values():
            parts.append(finding.to_markdown() + "\n\n---")

        return "\n\n".join(parts)


class InstitutionalSynthesizer:
    """Generates grounded sell-side research findings from curated evidence."""

    def __init__(self) -> None:
        pass

    def _build_evidence_context(self, evidence: Sequence[EvidenceChunk], max_chars_per_chunk: int = 450) -> str:
        """Formats top evidence chunks into a clean, human-readable, citable block."""
        blocks: list[str] = []
        for idx, chunk in enumerate(evidence[:3], 1):
            trail = f" | Section: {chunk.section_breadcrumb}" if chunk.section_breadcrumb else ""
            header = f"--- [EXHIBIT EVIDENCE {idx}] {chunk.citation_tag}{trail} ---"
            # Clean markdown table pipes and dialogue lines
            lines: list[str] = []
            for raw_line in chunk.content.split("\n"):
                line = raw_line.strip()
                if not line or set(line) <= {"-", "|", ":", " "}:
                    continue
                if "|" in line:
                    cells = [c.strip() for c in line.split("|") if c.strip() and not set(c.strip()) <= {"-", ":"}]
                    if cells:
                        line = " - ".join(cells)
                if re.match(r"^(?:moderator|operator|[A-Z][a-z]+ [A-Z][a-z]+):", line, re.IGNORECASE):
                    continue
                lines.append(line)

            clean_text = " ".join(" ".join(lines).split()) if lines else " ".join(chunk.content.split())
            if len(clean_text) > max_chars_per_chunk:
                clean_text = clean_text[:max_chars_per_chunk] + "..."
            blocks.append(f"{header}\n{clean_text}")
        return "\n\n".join(blocks)

    def synthesize_pillar(
        self,
        ticker: str,
        pillar: str,
        title: str,
        evidence: Sequence[EvidenceChunk],
    ) -> ThematicFinding:
        """Synthesizes qualitative evidence for a single research pillar with a high-density prompt."""
        sym = safe_ticker(ticker)
        if not evidence:
            logger.warning("[%s] No evidence provided for pillar '%s'.", sym, pillar)
            return ThematicFinding(
                pillar=pillar,
                title=title,
                bullet_points=["No primary disclosures found in uploaded filings for this topic."],
                citations=[],
            )

        context_text = self._build_evidence_context(evidence, max_chars_per_chunk=450)
        citations_list = [c.citation_tag for c in evidence[:3]]

        prompt = f"""You are a Senior Institutional Equity Research Analyst (Motilal Oswal / Kotak Equities style).
Analyze the primary filing evidence for {sym} and generate 2 to 3 clean, simple, and straight-to-the-point research bullet points regarding:
THEME: {title}

TASK RULES:
1. Keep every bullet SIMPLE, DIRECT, and STRAIGHT TO THE POINT. Focus purely on company financials, revenue drivers, margins, cash flows, deal momentum, capex, and key audit matters.
2. DO NOT write meta-commentary (e.g. NEVER write 'the extract does not quantify' or 'the cited page provides no data'). State what IS disclosed directly in clear business English.
3. NEVER mention speaker names, conversational dialogue, or operator pleasantries (e.g. Do NOT write 'Srini discussed' or 'operator stated').
4. Every bullet must start with a bold category lead (e.g. "- **[Revenue Driver / Margin Lever / Cash Flow / Key Audit Matter]**: ...").
5. Include concrete figures, percentages, rupee amounts (₹ Cr), and basis points (bps) from the context.
6. Append the exact short citation tag from the evidence block at the end of each bullet point (e.g. `(Investor Presentation, p. 5)`).

OUTPUT FORMAT:
- **[Category 1]**: [Direct, easily understandable business finding with facts/numbers] (Source Tag)
- **[Category 2]**: [Direct, easily understandable business finding with facts/numbers] (Source Tag)
- **[Category 3]**: [Direct, easily understandable business finding with facts/numbers] (Source Tag)
"""

        context_chars = len(context_text)
        prompt_chars = len(prompt)
        total_payload_chars = context_chars + prompt_chars
        approx_tokens = total_payload_chars // 4

        logger.info(
            "[%s] Pillar '%s' Metrics -> Chunks: %d | Context: %d chars | Total: %d chars (~%d tokens)",
            sym,
            pillar,
            len(evidence[:3]),
            context_chars,
            total_payload_chars,
            approx_tokens,
        )

        raw_response = ""
        last_error: Optional[Exception] = None

        for attempt in range(3):
            try:
                logger.info("[%s] Querying hosted LLM for '%s' (attempt %d/3)...", sym, pillar, attempt + 1)
                t0 = time.perf_counter()
                raw_response = generate_llm_response(prompt=prompt, context=context_text)
                lat = time.perf_counter() - t0
                if raw_response:
                    logger.info(
                        "[%s] Pillar '%s' LLM Success: %d chars in %.2fs.",
                        sym,
                        pillar,
                        len(raw_response),
                        lat,
                    )
                    break
            except Exception as exc:
                last_error = exc
                logger.warning("[%s] LLM attempt %d failed for pillar '%s': %s", sym, attempt + 1, pillar, exc)
                time.sleep(2.0 * (attempt + 1))

        if not raw_response:
            logger.error("[%s] LLM generation permanently failed for pillar '%s': %s", sym, pillar, last_error)
            first_c = evidence[0]
            clean_lines = [
                line.strip() for line in first_c.content.split("\n")
                if line.strip() and not line.strip().startswith("|") and not re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+:", line.strip())
            ]
            snip = " ".join(clean_lines)[:250] or " ".join(first_c.content.split())[:250]
            return ThematicFinding(
                pillar=pillar,
                title=title,
                bullet_points=[f"**Key Filing Disclosure**: {snip}... {first_c.citation_tag}"],
                citations=citations_list,
                raw_synthesis=str(last_error),
                chunks_used=[c.to_dict() for c in evidence[:3]],
            )

        # Parse bullets from LLM response
        bullets: list[str] = []
        for line in raw_response.strip().split("\n"):
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("-") or line_str.startswith("•") or re.match(r"^\d+\.", line_str):
                clean_bullet = re.sub(r"^[-•\d\.]+\s*", "", line_str).strip()
                if clean_bullet:
                    bullets.append(clean_bullet)

        if not bullets and raw_response:
            bullets = [raw_response.strip()]

        return ThematicFinding(
            pillar=pillar,
            title=title,
            bullet_points=bullets,
            citations=citations_list,
            raw_synthesis=raw_response,
            chunks_used=[c.to_dict() for c in evidence[:3]],
        )

    def synthesize_all(
        self,
        ticker: str,
        reranked_evidence: dict[str, Sequence[EvidenceChunk]],
        probes: Sequence[ThematicProbe],
        company_name: str = "",
        available_doc_types: Optional[Sequence[str]] = None,
        max_workers: int = 1,
    ) -> ResearchDossier:
        """Synthesizes findings across all research pillars with sequential pacing for 100% gateway reliability."""
        start_time = time.perf_counter()
        sym = safe_ticker(ticker)
        pillars_map: dict[str, ThematicFinding] = {}

        logger.info(
            "[%s] Reliable LLM Synthesis: synthesizing %d research pillars sequentially...",
            sym,
            len(probes),
        )

        ordered_probes = sorted(probes, key=lambda x: x.priority)
        for idx, probe in enumerate(ordered_probes, 1):
            evidence = reranked_evidence.get(probe.pillar, [])
            logger.info("[%s] [%d/%d] Synthesizing pillar '%s' (%s)...", sym, idx, len(probes), probe.pillar, probe.title)
            finding = self.synthesize_pillar(
                ticker=sym,
                pillar=probe.pillar,
                title=probe.title,
                evidence=evidence,
            )
            pillars_map[probe.pillar] = finding
            # Micro-pause between calls to prevent Vercel gateway throttling
            if idx < len(ordered_probes):
                time.sleep(0.5)

        # Preserve probe priority order
        ordered_pillars = {
            p.pillar: pillars_map[p.pillar]
            for p in sorted(probes, key=lambda x: x.priority)
            if p.pillar in pillars_map
        }

        elapsed = time.perf_counter() - start_time
        success_count = sum(1 for f in ordered_pillars.values() if not f.bullet_points[0].startswith("Synthesis error"))

        logger.info(
            "[%s] Concurrent Synthesis Complete: %d/%d pillars generated in %.2fs (avg %.2fs/pillar).",
            sym,
            success_count,
            len(probes),
            elapsed,
            elapsed / max(1, len(probes)),
        )

        return ResearchDossier(
            ticker=sym,
            company_name=company_name or sym,
            generated_at=utc_now(),
            pillars=ordered_pillars,
            available_doc_types=list(available_doc_types or []),
            summary=f"Institutional qualitative findings for {sym} synthesized from primary filings.",
        )
