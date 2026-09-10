"""Institutional Equity Research Probes for GrowNXT RAG Pipeline.

Defines sell-side research probe templates (Motilal Oswal, Kotak Equities, ICICI Direct style)
with adaptive document routing that handles tickers with missing transcripts or presentations.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

PILLAR_STRATEGY_GROWTH: str = "strategy_growth"
PILLAR_SEGMENT_DYNAMICS: str = "segment_dynamics"
PILLAR_MARGIN_COST: str = "margin_cost"
PILLAR_CAPITAL_ALLOCATION: str = "capital_allocation"
PILLAR_CONCALL_HIGHLIGHTS: str = "concall_highlights"
PILLAR_KEY_RISKS_AUDIT: str = "key_risks_audit"

PILLAR_TITLES: dict[str, str] = {
    PILLAR_STRATEGY_GROWTH: "Strategic Growth Pillars & Multi-Year Roadmap",
    PILLAR_SEGMENT_DYNAMICS: "Core Business Segments & Revenue Drivers",
    PILLAR_MARGIN_COST: "Operating Margins, Pricing Power & Cost Pressures",
    PILLAR_CAPITAL_ALLOCATION: "Capital Allocation, Working Capital & Balance Sheet Health",
    PILLAR_CONCALL_HIGHLIGHTS: "Earnings Concall Highlights & Management Guidance",
    PILLAR_KEY_RISKS_AUDIT: "Key Audit Matters, Regulatory Disclosures & Risk Factors",
}


@dataclass
class ThematicProbe:
    """A research probe targeting a specific institutional equity research theme."""

    pillar: str
    title: str
    queries: list[str]
    target_doc_types: list[str] = field(default_factory=list)
    fallback_doc_types: list[str] = field(default_factory=lambda: ["annual_report"])
    priority: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "pillar": self.pillar,
            "title": self.title,
            "queries": list(self.queries),
            "target_doc_types": list(self.target_doc_types),
            "fallback_doc_types": list(self.fallback_doc_types),
            "priority": self.priority,
        }


def get_default_probes() -> list[ThematicProbe]:
    """Returns the comprehensive master suite of institutional research probes."""
    return [
        ThematicProbe(
            pillar=PILLAR_STRATEGY_GROWTH,
            title=PILLAR_TITLES[PILLAR_STRATEGY_GROWTH],
            queries=[
                "management strategic vision multi-year revenue growth targets and expansion roadmap",
                "capital expenditure commitments capex projects digital transformation and AI investments",
                "organic growth drivers new market expansion client acquisitions and deal pipeline",
            ],
            target_doc_types=["concall_presentation", "presentation", "annual_report"],
            priority=1,
        ),
        ThematicProbe(
            pillar=PILLAR_SEGMENT_DYNAMICS,
            title=PILLAR_TITLES[PILLAR_SEGMENT_DYNAMICS],
            queries=[
                "core business segments vertical revenue contribution and geographic mix",
                "key operating divisions product lines and service offerings breakdown",
                "segment performance growth momentum and market share trajectory",
            ],
            target_doc_types=["annual_report", "concall_presentation", "presentation"],
            priority=2,
        ),
        ThematicProbe(
            pillar=PILLAR_MARGIN_COST,
            title=PILLAR_TITLES[PILLAR_MARGIN_COST],
            queries=[
                "operating EBITDA EBIT margin trajectory levers and efficiency improvement programs",
                "cost headwinds raw material wage inflation pricing power and contract renegotiations",
                "SG&A leverage operational efficiency and profitability bridge",
            ],
            target_doc_types=["concall_transcript", "transcript", "annual_report"],
            priority=3,
        ),
        ThematicProbe(
            pillar=PILLAR_CAPITAL_ALLOCATION,
            title=PILLAR_TITLES[PILLAR_CAPITAL_ALLOCATION],
            queries=[
                "free cash flow generation working capital cycle days and cash conversion",
                "debt repayment schedule net debt to equity and liquidity position",
                "capital allocation policy dividend payout share buyback and M&A deployment",
            ],
            target_doc_types=["annual_report", "concall_transcript", "transcript"],
            priority=4,
        ),
        ThematicProbe(
            pillar=PILLAR_CONCALL_HIGHLIGHTS,
            title=PILLAR_TITLES[PILLAR_CONCALL_HIGHLIGHTS],
            queries=[
                "management commentary demand environment order book and qualitative guidance",
                "analyst questions pricing pressure margin headwinds and executive responses",
                "key takeaways from latest quarterly earnings conference call Q&A",
            ],
            target_doc_types=["concall_transcript", "transcript"],
            fallback_doc_types=["annual_report"],
            priority=5,
        ),
    ]


def build_adaptive_probes(
    ticker: str,  # noqa: ARG001 - kept for call-site/interface compatibility
    available_doc_types: Sequence[str],
) -> list[ThematicProbe]:
    """Builds adaptive research probes based on document availability for a stock.

    If a ticker lacks concall transcripts or presentations, probes automatically
    re-route target documents to the Annual Report's MD&A and Director's Report,
    ensuring consistent research depth without failing on missing filing classes.

    Args:
        ticker: Stock symbol (e.g. WIPRO, TCS, HDFCBANK).
        available_doc_types: Sequence of available doc_type strings (e.g. ['annual_report']).

    Returns:
        list[ThematicProbe]: Configured and adapted research probes.
    """
    avail_set = {d.strip().lower() for d in available_doc_types}
    master_probes = get_default_probes()
    adaptive_probes: list[ThematicProbe] = []

    has_transcript = any(t in avail_set for t in ("concall_transcript", "transcript"))
    has_annual = "annual_report" in avail_set or len(avail_set) == 0

    for probe in master_probes:
        # Check if any target doc type is available
        matched_targets = [dt for dt in probe.target_doc_types if dt in avail_set]

        if matched_targets:
            adaptive_probes.append(
                ThematicProbe(
                    pillar=probe.pillar,
                    title=probe.title,
                    queries=probe.queries,
                    target_doc_types=matched_targets,
                    priority=probe.priority,
                )
            )
        elif probe.pillar == PILLAR_CONCALL_HIGHLIGHTS and not has_transcript:
            # Fallback when transcript is missing: query Annual Report MD&A / Director's Report
            adapted_queries = [
                "management discussion and analysis business outlook and strategic guidance",
                "director report operational highlights industry environment and future prospects",
                "management commentary on performance drivers and market conditions",
            ]
            adaptive_probes.append(
                ThematicProbe(
                    pillar=probe.pillar,
                    title="Management Discussion & Strategic Outlook (Annual Report MD&A)",
                    queries=adapted_queries,
                    target_doc_types=["annual_report"]
                    if has_annual
                    else list(avail_set),
                    priority=probe.priority,
                )
            )
        else:
            # General fallback to available doc types
            fallback = [
                dt for dt in probe.fallback_doc_types if dt in avail_set
            ] or list(avail_set)
            adaptive_probes.append(
                ThematicProbe(
                    pillar=probe.pillar,
                    title=probe.title,
                    queries=probe.queries,
                    target_doc_types=fallback,
                    priority=probe.priority,
                )
            )

    return sorted(adaptive_probes, key=lambda p: p.priority)
