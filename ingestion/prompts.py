"""Extraction prompts for institutional-grade report generation.

One template serves all three document classes. What differs between an annual
report, an earnings call, and an investor deck is not the task -- extract what
a research analyst would carry forward -- but what each source can be trusted
to establish, so the class-specific part of the prompt is a *profile*: what the
document is authoritative for, what it is not, which topics to look for, and
how a claim from it must be attributed.

Everything else in the prompt is assembled from the document actually in hand.
A static prompt asks an annual report about its risk section whether or not one
was found, invites the model to fill the gap, and gets a fabricated answer that
looks exactly like a real one. So the targets are filtered to the sections the
parser actually produced, the retrieval probes carry the company's own name and
period, and the evidence block is numbered so that every finding must cite the
chunk it came from.

The output is a JSON contract rather than prose. Findings feed the report
generator's sections, and a finding whose figures cannot be traced to a chunk
is worse than a missing one: it survives into a typeset report carrying the
same authority as a verified figure.

Google Python Style Guide Compliant.
"""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ingestion.chunker import Chunk
from ingestion.registry import (
    DOC_TYPE_ANNUAL_REPORT,
    DOC_TYPE_PRESENTATION,
    DOC_TYPE_TRANSCRIPT,
)

# v2: probe queries carry the topic only, not the chunk's context header.
PROMPT_VERSION: str = "extraction-prompt/v2"

# Sections of the institutional report that findings can feed. These match the
# report generator's own section keys, so a finding routes without translation.
REPORT_SECTIONS: List[Tuple[str, str]] = [
    ("company_overview", "Executive summary and corporate profile"),
    ("company_operations", "Business segments and revenue engine"),
    ("expansion_plans", "Strategic expansion and capital allocation"),
    ("clients_market", "Competitive moat, clients and market footprint"),
    ("financial_results", "Financial performance and growth"),
    ("dupont_analysis", "Return decomposition, ROE and ROCE drivers"),
    ("balance_sheet", "Capital structure, solvency and liquidity"),
    ("strengths_weaknesses", "Investment thesis, risks and red flags"),
]

REPORT_SECTION_KEYS: List[str] = [key for key, _ in REPORT_SECTIONS]


@dataclass
class FocusArea:
    """One thing to look for in a document, and how to retrieve it.

    Attributes:
        focus_id: Stable key, used to group findings and cache probes.
        title: What an analyst would call this line of enquiry.
        probe: Retrieval query, embedded and matched against chunk vectors.
        sections: Parser section keys this area is expected to live in. Empty
            means any section. Used to skip areas whose section is absent, and
            to bias retrieval towards the right part of the document.
        report_sections: Report sections the findings feed.
    """

    focus_id: str
    title: str
    probe: str
    sections: List[str] = field(default_factory=list)
    report_sections: List[str] = field(default_factory=list)


@dataclass
class DocProfile:
    """What one document class can and cannot establish.

    Attributes:
        doc_type: Document class key.
        display: Name used in prompts.
        authority: What this source establishes, stated for the model.
        limits: What it must not be read as establishing.
        attribution: Default attribution for claims drawn from it.
        focus_areas: Lines of enquiry for this class.
    """

    doc_type: str
    display: str
    authority: str
    limits: str
    attribution: str
    focus_areas: List[FocusArea] = field(default_factory=list)


# --- Profiles -----------------------------------------------------------------

_ANNUAL_REPORT_PROFILE = DocProfile(
    doc_type=DOC_TYPE_ANNUAL_REPORT,
    display="Integrated Annual Report",
    authority=(
        "An audited, board-approved filing. Figures in the financial statements "
        "and the segment tables are the company's official record for the "
        "financial year, and management's discussion is its own explanation of "
        "those figures. Statements about strategy, capital allocation, and risk "
        "are formal disclosures the company is accountable for."
    ),
    limits=(
        "It is retrospective and written to be favourable. The narrative "
        "sections select which facts to emphasise, forward-looking statements "
        "are aspirations rather than guidance, and the notice of meeting, the "
        "governance report, and the ESG report are compliance documents that "
        "say little about business performance."
    ),
    attribution="company_disclosure",
    focus_areas=[
        FocusArea(
            focus_id="business_model",
            title="Business model and revenue engine",
            probe=(
                "how the company earns revenue, its business segments, service "
                "lines, and the share of revenue each contributes"
            ),
            sections=["business", "mdna"],
            report_sections=["company_overview", "company_operations"],
        ),
        FocusArea(
            focus_id="segment_performance",
            title="Segment and geography performance",
            probe=(
                "revenue and margin by business segment and by geography, "
                "year-on-year growth of each segment"
            ),
            sections=["mdna", "business", "financials"],
            report_sections=["company_operations", "financial_results"],
        ),
        FocusArea(
            focus_id="operating_results",
            title="Results of operations and margins",
            probe=(
                "results of operations, revenue growth, operating margin, "
                "earnings per share, and the reasons management gives for the "
                "change against the prior year"
            ),
            sections=["mdna", "financials"],
            report_sections=["financial_results", "dupont_analysis"],
        ),
        FocusArea(
            focus_id="capital_allocation",
            title="Capital allocation and expansion",
            probe=(
                "capital expenditure, acquisitions, new facilities, dividends, "
                "buybacks, and how the company intends to deploy capital"
            ),
            sections=["mdna", "business", "governance"],
            report_sections=["expansion_plans", "balance_sheet"],
        ),
        FocusArea(
            focus_id="liquidity_solvency",
            title="Liquidity, borrowings and cash generation",
            probe=(
                "cash and cash equivalents, borrowings, net debt, operating "
                "cash flow, free cash flow, and credit facilities"
            ),
            sections=["mdna", "financials"],
            report_sections=["balance_sheet", "financial_results"],
        ),
        FocusArea(
            focus_id="clients_moat",
            title="Clients, contracts and competitive position",
            probe=(
                "client concentration, largest customers, order book, long-term "
                "contracts, and the company's stated competitive advantages"
            ),
            sections=["business", "mdna"],
            report_sections=["clients_market", "company_operations"],
        ),
        FocusArea(
            focus_id="risks",
            title="Risk factors and mitigations",
            probe=(
                "principal risks, risk factors, concentration risk, regulatory "
                "and currency exposure, and the mitigations described"
            ),
            sections=["risk", "mdna", "governance"],
            report_sections=["strengths_weaknesses"],
        ),
        FocusArea(
            focus_id="accounting_flags",
            title="Accounting policies and auditor observations",
            probe=(
                "auditor's opinion, key audit matters, changes in accounting "
                "policy, contingent liabilities, and related party transactions"
            ),
            sections=["financials", "governance"],
            report_sections=["strengths_weaknesses", "balance_sheet"],
        ),
    ],
)

_TRANSCRIPT_PROFILE = DocProfile(
    doc_type=DOC_TYPE_TRANSCRIPT,
    display="Earnings Call Transcript",
    authority=(
        "The most current management commentary available, and the only source "
        "where management is questioned adversarially. Guidance, deal wins, "
        "demand commentary, and margin trajectory are stated here first. "
        "Analyst questions are direct evidence of what the market doubts, and "
        "an evaded question is itself a finding."
    ),
    limits=(
        "It is unaudited speech. Figures quoted on a call are approximate and "
        "may be restated in the filing, so they must never override the annual "
        "report's audited figures. Management's characterisation of a trend is "
        "an opinion, and must be attributed as one."
    ),
    attribution="management_statement",
    focus_areas=[
        FocusArea(
            focus_id="guidance",
            title="Guidance and outlook",
            probe=(
                "revenue guidance for the coming quarter and year, margin "
                "outlook, and the assumptions management attaches to it"
            ),
            sections=["prepared_remarks", "qa"],
            report_sections=["financial_results", "strengths_weaknesses"],
        ),
        FocusArea(
            focus_id="quarter_performance",
            title="Quarter performance and drivers",
            probe=(
                "revenue and margin for the quarter, sequential and "
                "year-on-year change, and what management says drove it"
            ),
            sections=["prepared_remarks"],
            report_sections=["financial_results"],
        ),
        FocusArea(
            focus_id="demand_environment",
            title="Demand environment by segment and geography",
            probe=(
                "demand commentary by industry vertical and region, deal "
                "pipeline, order booking, large deal wins, and client budgets"
            ),
            sections=["prepared_remarks", "qa"],
            report_sections=["company_operations", "clients_market"],
        ),
        FocusArea(
            focus_id="margin_levers",
            title="Margin levers and cost actions",
            probe=(
                "margin drivers, utilisation, pricing, wage hikes, headcount, "
                "attrition, and cost efficiency programmes"
            ),
            sections=["prepared_remarks", "qa"],
            report_sections=["financial_results", "dupont_analysis"],
        ),
        FocusArea(
            focus_id="capital_returns",
            title="Capital allocation and shareholder returns",
            probe=(
                "dividend, buyback, acquisitions, capital expenditure plans, "
                "and cash deployment discussed on the call"
            ),
            sections=["prepared_remarks", "qa"],
            report_sections=["expansion_plans", "balance_sheet"],
        ),
        FocusArea(
            focus_id="analyst_concerns",
            title="Analyst concerns and management's answers",
            probe=(
                "questions analysts pressed on repeatedly, scepticism about "
                "growth or margins, and where management declined to answer"
            ),
            sections=["qa"],
            report_sections=["strengths_weaknesses"],
        ),
        FocusArea(
            focus_id="strategy_shifts",
            title="Strategy shifts and new initiatives",
            probe=(
                "new strategy, reorganisation, leadership change, new products "
                "or platforms, and investments in new capability"
            ),
            sections=["prepared_remarks", "qa"],
            report_sections=["expansion_plans", "company_operations"],
        ),
    ],
)

_PRESENTATION_PROFILE = DocProfile(
    doc_type=DOC_TYPE_PRESENTATION,
    display="Investor Presentation",
    authority=(
        "Management's own framing of the numbers, and usually the cleanest "
        "source for segment splits, client metrics, and multi-year series that "
        "the filings report only in fragments."
    ),
    limits=(
        "It is a selling document. Metrics are chosen and time windows are "
        "picked to flatter, non-standard measures appear without reconciliation, "
        "and charts carry no axis when the axis would be unhelpful. Nothing here "
        "is audited, and an absent metric is often the informative one."
    ),
    attribution="company_presentation",
    focus_areas=[
        FocusArea(
            focus_id="headline_metrics",
            title="Headline metrics and multi-year series",
            probe=(
                "revenue, margin, headcount and cash figures presented for the "
                "quarter and across years"
            ),
            sections=["slides"],
            report_sections=["financial_results"],
        ),
        FocusArea(
            focus_id="segment_mix",
            title="Segment, vertical and geography mix",
            probe=(
                "revenue split by business segment, industry vertical, service "
                "line and geography, with the share each contributes"
            ),
            sections=["slides"],
            report_sections=["company_operations"],
        ),
        FocusArea(
            focus_id="client_metrics",
            title="Client and deal metrics",
            probe=(
                "number of clients, client concentration, million-dollar "
                "relationships, order booking and total contract value"
            ),
            sections=["slides"],
            report_sections=["clients_market"],
        ),
        FocusArea(
            focus_id="strategy_slides",
            title="Strategy and investment priorities",
            probe=(
                "strategic priorities, target markets, new capability "
                "investment and the medium-term ambition presented"
            ),
            sections=["slides"],
            report_sections=["expansion_plans", "company_overview"],
        ),
        FocusArea(
            focus_id="non_standard_measures",
            title="Non-standard measures and presentation choices",
            probe=(
                "adjusted, normalised, constant currency and like-for-like "
                "measures, restated figures and excluded items"
            ),
            sections=["slides"],
            report_sections=["strengths_weaknesses", "financial_results"],
        ),
    ],
)

DOC_PROFILES: Dict[str, DocProfile] = {
    DOC_TYPE_ANNUAL_REPORT: _ANNUAL_REPORT_PROFILE,
    DOC_TYPE_TRANSCRIPT: _TRANSCRIPT_PROFILE,
    DOC_TYPE_PRESENTATION: _PRESENTATION_PROFILE,
}


# --- Retrieval probes ---------------------------------------------------------

def build_probes(
    doc_type: str,
    company: str,
    label: str,
    available_sections: Optional[Iterable[str]] = None,
) -> List[Tuple[FocusArea, str]]:
    """Builds the retrieval queries for one document.

    The query is the topic and nothing else. Context belongs on the chunk, not
    on both sides of the comparison: every chunk's embedding text opens with the
    same header -- company, document class, period, section -- and a query that
    repeats it scores that header against itself for every candidate. The chunks
    that win such a comparison are the ones with the least content, because a
    near-empty chunk *is* its header. That is not a hypothetical: with the
    company and period in the query, the top match for "revenue and margin by
    business segment" in a large-cap annual report was a chunk whose entire
    text was ``2 2``, at 0.922, ahead of every real segment disclosure.

    The document is already narrowed to one filing before any probe runs, and
    the section filter narrows it further, so the company and period add nothing
    the search does not already know.

    Focus areas whose sections are all absent are dropped rather than searched:
    retrieving the nearest chunks to "principal risks" from a document with no
    risk section returns the closest thing available, which is then presented as
    if it were the risk disclosure.

    Args:
        doc_type: Document class.
        company: Company name, or the ticker when the name is unknown. Retained
            for the caller's logging and for the prompt header; deliberately not
            part of the query.
        label: Period label of the document, likewise.
        available_sections: Parser section keys present in the document.

    Returns:
        Pairs of (focus area, query text).
    """
    profile = DOC_PROFILES.get(doc_type)
    if profile is None:
        return []
    present = set(available_sections or ())
    probes: List[Tuple[FocusArea, str]] = []
    for area in profile.focus_areas:
        if present and area.sections and not (set(area.sections) & present):
            continue
        probes.append((area, "%s: %s" % (area.title, area.probe)))
    return probes


# --- Prompt assembly ----------------------------------------------------------

def _format_evidence(chunks: Sequence[Chunk], max_chars: int) -> Tuple[str, List[str]]:
    """Renders retrieved chunks as a numbered, citable evidence block.

    Args:
        chunks: Chunks to include, most relevant first.
        max_chars: Budget for the whole block.

    Returns:
        The rendered block and the chunk ids actually included. Chunks past the
        budget are dropped rather than truncated mid-figure, and the caller is
        told which survived so the prompt's citation list matches its evidence.
    """
    lines: List[str] = []
    included: List[str] = []
    used = 0
    for chunk in chunks:
        pages = ("p%d" % chunk.page_start if chunk.page_start == chunk.page_end
                 else "pp%d-%d" % (chunk.page_start, chunk.page_end))
        attribution = ""
        if chunk.speaker:
            attribution = " | %s (%s)" % (chunk.speaker, chunk.role or "speaker")
        elif chunk.title:
            attribution = " | %s" % chunk.title
        head = "[%s] %s | %s%s" % (chunk.chunk_id, chunk.section_title, pages, attribution)
        entry = head + "\n" + chunk.text
        if used + len(entry) > max_chars and included:
            break
        lines.append(entry)
        included.append(chunk.chunk_id)
        used += len(entry)
    return "\n\n".join(lines), included


def _output_contract() -> str:
    """Returns the JSON contract the model must answer with."""
    schema = {
        "doc_id": "<document id, copied from the header>",
        "findings": [
            {
                "focus_id": "<one of the focus ids listed above>",
                "topic": "<short label for this finding>",
                "claim": "<one or two sentences, in your own words>",
                "figures": [
                    {
                        "metric": "<what is measured>",
                        "value": "<exactly as printed in the evidence>",
                        "unit": "<currency, %, count, or empty>",
                        "period": "<the period the figure belongs to>",
                    }
                ],
                "evidence": ["<chunk_id>"],
                "attribution": "<company_disclosure | management_statement | "
                               "analyst_question | company_presentation>",
                "materiality": "<high | medium | low>",
                "report_sections": ["<report section keys this finding feeds>"],
                "confidence": "<0.0-1.0>",
                "caveat": "<what would change this reading, or empty>",
            }
        ],
        "absences": [
            "<a target that the evidence does not cover, named plainly>"
        ],
    }
    return json.dumps(schema, indent=2)


def build_extraction_prompt(
    ticker: str,
    company: str,
    doc_id: str,
    doc_type: str,
    label: str,
    chunks: Sequence[Chunk],
    focus_areas: Sequence[FocusArea],
    available_sections: Optional[Iterable[str]] = None,
    evidence_budget: int = 24000,
) -> Dict[str, Any]:
    """Assembles the extraction prompt for one document.

    Args:
        ticker: Stock ticker symbol.
        company: Company name.
        doc_id: Registry identifier of the document.
        doc_type: Document class.
        label: Period label.
        chunks: Retrieved chunks, most relevant first.
        focus_areas: Focus areas the retrieval actually covered.
        available_sections: Parser section keys present in the document.
        evidence_budget: Character budget for the evidence block.

    Returns:
        A mapping with the rendered ``prompt``, the ``chunk_ids`` cited in it,
        and the ``focus_ids`` it asks about, so that a caller can verify a
        response's citations against what it was actually shown.

    Raises:
        KeyError: If `doc_type` has no profile.
    """
    profile = DOC_PROFILES[doc_type]
    evidence, cited = _format_evidence(chunks, evidence_budget)
    sections = sorted(set(available_sections or ()))

    target_lines = []
    for area in focus_areas:
        feeds = ", ".join(area.report_sections) or "any"
        target_lines.append("- %s (focus_id: %s) -> feeds: %s\n  %s"
                            % (area.title, area.focus_id, feeds, area.probe))

    report_lines = ["- %s: %s" % (key, title) for key, title in REPORT_SECTIONS]

    prompt = """You are a senior equity research analyst preparing an institutional-grade
fundamental report on a listed company. Your task on this pass is extraction,
not writing: pull out every finding a report author would need, and nothing
that is not in the evidence.

=== COMPANY AND DOCUMENT ===
Company: {company} ({ticker})
Document: {display} - {label}
Document id: {doc_id}
Sections present in this document: {sections}
Evidence passages provided: {n_chunks}

=== WHAT THIS SOURCE ESTABLISHES ===
{authority}

=== WHAT IT DOES NOT ESTABLISH ===
{limits}

=== EXTRACTION TARGETS ===
{targets}

=== REPORT SECTIONS YOUR FINDINGS FEED ===
{report_sections}

=== EVIDENCE ===
Each passage is headed by its chunk id, section, and page range. Cite these ids.

{evidence}

=== RULES ===
1. Every finding must be supported by at least one chunk id from the evidence
   above. A finding you cannot cite does not go in the output.
2. Reproduce figures exactly as printed, including the currency, the scale
   ("million", "crore"), and brackets for negatives. Do not convert, round, or
   annualise. If a figure's period is not stated in the evidence, leave the
   period empty rather than inferring it.
3. Do not compute derived metrics. Ratios and growth rates are calculated
   elsewhere from the statutory data; a figure you arithmetic together here
   cannot be verified against it.
4. Attribute correctly. A claim by management is not a fact about the business;
   an analyst's question is evidence of market concern, not of the answer.
5. Distinguish what happened from what is intended. Guidance, targets, and
   plans are forward-looking and must read as such in the claim.
6. Name what is missing. If a target above has no support in the evidence, list
   it under "absences". An honest absence is more useful to the report than a
   thin inference, and far more useful than a plausible invention.
7. Prefer specific findings over comprehensive ones. Ten findings that each
   carry a figure and a citation beat thirty that restate the narrative.

=== OUTPUT ===
Return only a JSON object of this shape, with no commentary around it:

{contract}
""".format(
        company=company or ticker,
        ticker=ticker,
        display=profile.display,
        label=label,
        doc_id=doc_id,
        sections=", ".join(sections) or "unclassified",
        n_chunks=len(cited),
        authority=profile.authority,
        limits=profile.limits,
        targets="\n".join(target_lines) or "- (no target applies to this document)",
        report_sections="\n".join(report_lines),
        evidence=evidence or "(no evidence retrieved)",
        contract=_output_contract(),
    )

    return {
        "prompt": prompt,
        "prompt_version": PROMPT_VERSION,
        "doc_id": doc_id,
        "doc_type": doc_type,
        "chunk_ids": cited,
        "focus_ids": [area.focus_id for area in focus_areas],
        "default_attribution": profile.attribution,
    }
