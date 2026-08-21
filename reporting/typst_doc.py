"""Typst source generation for GrowNXT institutional reports.

Tables are emitted as native Typst markup rather than routed through an
intermediate format. Typst's `table` gives column spanners, per-column
alignment and explicit stroke control directly, so there is nothing to gain
from generating HTML or LaTeX first and everything to lose in fidelity.

Structural conventions, taken from how sell-side research is actually set:

    - Every table and chart is a numbered EXHIBIT with a caption above and
      a source line below. Numbering is what lets body text refer to a
      figure, and it is the clearest marker of research-grade structure.
    - Sections are SEMANTIC, not incidental: each answers one question, and
      an exhibit only appears in the section whose question it addresses.
      The margin table sits with the income statement it decomposes; the
      leverage series sits with the balance sheet it is drawn from; the
      cash cycle sits with earnings quality because both ask whether
      reported profit is real.
    - Content is paired two-up wherever both halves are narrow, so a page
      fills instead of leaving a column of white beside a short table.
    - Booktabs discipline: no vertical rules, three horizontal ones — above
      the header, below the header, below the body.
"""

from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from reporting import fmt, tokens
from reporting.analytics import DerivedAnalytics
from reporting.composites import Composites
from reporting.selfcheck import SelfCheck
from reporting.snapshot import CompanySnapshot

# Typst literals for the token palette.
C_INK = 'rgb("%s")' % tokens.INK
C_INK_SOFT = 'rgb("%s")' % tokens.INK_SOFT
C_MUTED = 'rgb("%s")' % tokens.MUTED
C_RULE = 'rgb("%s")' % tokens.RULE
C_BRAND = 'rgb("%s")' % tokens.BRAND
C_BRAND_TINT = 'rgb("%s")' % tokens.BRAND_TINT
C_SUNKEN = 'rgb("%s")' % tokens.SUNKEN
C_SUNKEN_DEEP = 'rgb("%s")' % tokens.SUNKEN_DEEP
C_POS = 'rgb("%s")' % tokens.POSITIVE
C_NEG = 'rgb("%s")' % tokens.NEGATIVE

SOURCE_DEFAULT = "Financial Data Collector; GrowNXT computations"

# Tables use the same display window as the charts, so an exhibit and the
# figure beside it always cover identical periods.
TABLE_PERIODS: int = tokens.DISPLAY_YEARS


class Exhibits:
    """Sequential exhibit numbering for one document."""

    def __init__(self) -> None:
        """Starts numbering at one."""
        self._next = 1

    def caption(self, title: str) -> str:
        """Renders a numbered exhibit caption, sticky to what follows."""
        number = self._next
        self._next += 1
        return (
            "#block(sticky: true, above: 7pt, below: 3pt)[\n"
            "  #text(size: %s, weight: \"semibold\", fill: %s)[Exhibit %d.]"
            " #text(size: %s, fill: %s)[%s]\n"
            "]\n" % (
                tokens.SIZE_H2, C_BRAND, number,
                tokens.SIZE_H2, C_INK, fmt.escape_typst(title),
            )
        )

    def source(self, note: str = SOURCE_DEFAULT) -> str:
        """Renders the source line that closes an exhibit."""
        return "#block(above: 2.5pt, below: 5pt)[#text(size: %s, fill: %s)[Source: %s]]\n" % (
            tokens.SIZE_FOOTNOTE, C_MUTED, fmt.escape_typst(note),
        )

    def wrap(self, title: str, body: str, note: str = SOURCE_DEFAULT) -> str:
        """Renders caption, body and source as one non-breakable unit.

        Keeping the three together is what stops a source line being pushed
        alone onto the next page, orphaned from the exhibit it describes.
        """
        return ("#block(breakable: false, width: 100%)[\n"
                + self.caption(title) + body + self.source(note)
                + "]\n")


def _font_list(fonts: Sequence[str]) -> str:
    """Renders a font fallback chain as a Typst array literal."""
    return "(" + ", ".join('"%s"' % f for f in fonts) + ")"


def _cell(text: str) -> str:
    """Wraps a pre-formatted string as a Typst content cell."""
    return "[" + text + "]"


def _bold_cell(text: str) -> str:
    """Wraps a cell and sets it semibold, for summary rows."""
    return "[#text(weight: \"semibold\")[" + text + "]]"


def _delta_cell(value: Optional[float], decimals: int = 1) -> str:
    """Renders a growth figure, coloured by direction.

    Colour here encodes direction only — the one semantic use permitted by
    the palette policy.
    """
    if value is None:
        return _cell(fmt.DASH)
    colour = C_POS if value >= 0 else C_NEG
    return "[#text(fill: %s)[%s]]" % (colour, fmt.signed_pct(value, decimals))


def _residual_cell(value: Optional[float]) -> str:
    """Renders a self-check residual, de-emphasised when it is nil.

    A residual that closes is the expected case and should recede; one that
    does not must be the most visible thing in the row. Printing a hard zero
    for a value that is merely small would be a claim the arithmetic cannot
    support, so anything below display precision renders as a true nil glyph
    and anything above it renders in full and in the alert colour.
    """
    if value is None:
        return _cell(fmt.DASH)
    if abs(value) < 5e-5:
        return "[#text(fill: %s)[nil]]" % C_MUTED
    return "[#text(fill: %s, weight: \"semibold\")[%s]]" % (
        C_NEG, fmt.signed_pct(value, 4))


def _verdict_cell(passed: Optional[bool], points: Optional[int]) -> str:
    """Renders a scored sub-test's point award, coloured by direction.

    The digit is kept rather than replaced with a tick because the column has
    to be seen to sum to the total printed beneath it. A reader who cannot
    add the column up has been given a score to trust rather than to check.
    """
    if passed is None or points is None:
        return "[#text(fill: %s)[%s]]" % (C_MUTED, fmt.DASH)
    colour = C_POS if passed else C_NEG
    return "[#text(fill: %s, weight: \"semibold\")[%d]]" % (colour, points)


def _score_value(value: Optional[float], unit: str) -> str:
    """Formats a sub-test figure according to the unit it is measured in."""
    if unit == "pct":
        return fmt.pct(value, 2)
    if unit == "cr":
        return fmt.num(value)
    if unit == "x":
        return fmt.mult(value)
    return fmt.ratio(value)


def _note_block(text: str, tone: str = "neutral") -> str:
    """Renders an indented note beneath an exhibit.

    Used where a figure needs a caveat that is too long for a source line and
    too important to omit: a withheld framework, a substituted input, a score
    dominated by one term.
    """
    stripe = C_BRAND if tone == "neutral" else C_NEG
    return (
        "#block(inset: (left: 5pt), above: 5.5pt, below: 5pt, width: 100%%, "
        "stroke: (left: 1.4pt + %s))[\n"
        "  #text(size: %s, fill: %s)[%s]\n"
        "]\n" % (stripe, tokens.SIZE_SMALL, C_INK_SOFT, fmt.escape_typst(text))
    )


def _table(
    columns: str,
    align: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    size: str = tokens.SIZE_TABLE,
    emphasise_last: bool = False,
) -> str:
    """Builds a booktabs-styled Typst table.

    Args:
        columns: Typst column spec, e.g. '(auto, 1fr, 1fr)'.
        align: Typst alignment spec matching the column count.
        header: Header cell contents, already escaped.
        rows: Body rows of pre-formatted cell strings.
        size: Text size for the table.
        emphasise_last: Rule off and embolden the final row, for a summary
            period such as TTM or a bridge total.

    Returns:
        Typst markup, or an italic note when there are no rows — an empty
        table frame is worse than an explicit absence.
    """
    if not rows:
        return "#text(size: %s, fill: %s)[_Not reported by the data provider._]\n" % (
            tokens.SIZE_SMALL, C_MUTED,
        )

    parts = [
        "#block(breakable: false, width: 100%)[",
        "#set text(size: %s)" % size,
        "#table(",
        "  columns: %s," % columns,
        "  align: %s," % align,
        "  stroke: none,",
        "  inset: (x: 3.6pt, y: 2.5pt),",
        "  fill: (_, y) => if y == 0 { %s } else if calc.even(y) { %s } else { none },"
        % (C_SUNKEN_DEEP, C_SUNKEN),
        "  table.hline(y: 0, stroke: 0.7pt + %s)," % C_INK,
        "  table.header(%s)," % ", ".join(
            "[#text(fill: %s, weight: \"semibold\")[%s]]" % (C_INK_SOFT, h) for h in header
        ),
        "  table.hline(y: 1, stroke: 0.4pt + %s)," % C_RULE,
    ]
    for index, row in enumerate(rows):
        if emphasise_last and index == len(rows) - 1:
            parts.append("  table.hline(stroke: 0.4pt + %s)," % C_RULE)
            parts.append("  " + ", ".join(
                cell if cell.startswith("[#text(weight") else
                "[#text(weight: \"semibold\")" + cell + "]"
                for cell in row
            ) + ",")
        else:
            parts.append("  " + ", ".join(row) + ",")
    parts.append("  table.hline(stroke: 0.7pt + %s)," % C_INK)
    parts.append(")")
    parts.append("]")
    return "\n".join(parts) + "\n"


def _section(title: str, question: str = "") -> str:
    """Renders a section heading with a brand rule beneath it.

    Args:
        title: Section title.
        question: Optional one-line statement of what the section answers,
            which is what keeps the grouping legible to a reader who is
            scanning rather than reading.
    """
    subtitle = ""
    if question:
        subtitle = ("  #v(1pt)\n  #text(size: %s, fill: %s)[%s]\n"
                    % (tokens.SIZE_SMALL, C_MUTED, fmt.escape_typst(question)))
    return (
        "\n#block(above: 11pt, below: 4pt, sticky: true)[\n"
        "  #text(font: %s, size: %s, weight: \"bold\", fill: %s)[%s]\n"
        "  #v(-3.5pt)\n"
        "  #line(length: 100%%, stroke: 0.8pt + %s)\n"
        "%s"
        "]\n" % (
            _font_list(tokens.FONT_DISPLAY), tokens.SIZE_H1, C_INK,
            fmt.escape_typst(title), C_BRAND, subtitle,
        )
    )


def _two_up(left: str, right: str, ratio: str = "(1fr, 1fr)") -> str:
    """Places two blocks side by side, top-aligned.

    Top alignment matters: with the default, a short table beside a tall
    chart is vertically centred and the pairing reads as accidental.
    """
    return (
        "#grid(columns: %s, gutter: %s, align: top,\n  [%s],\n  [%s],\n)\n"
        % (ratio, tokens.GUTTER, left, right)
    )


def _image(path: Path, width: str = "100%") -> str:
    """Embeds a chart SVG by bare filename."""
    return "#image(\"%s\", width: %s)" % (path.name, width)


def _panel(title: str, pairs: Sequence[Sequence[str]]) -> str:
    """Renders a titled key/value panel for the cover tearsheet."""
    rows = []
    for label, value in pairs:
        rows.append(
            "    [#text(fill: %s)[%s]], [#text(weight: \"semibold\")[%s]],"
            % (C_MUTED, fmt.escape_typst(label), value)
        )
    return "\n".join([
        "#block(fill: %s, inset: (x: 6pt, y: 5pt), radius: 1.5pt, width: 100%%, "
        "stroke: (top: 1.2pt + %s))[" % (C_SUNKEN, C_BRAND),
        "  #text(size: %s, weight: \"semibold\", fill: %s, tracking: 0.4pt)[%s]" % (
            tokens.SIZE_FOOTNOTE, C_BRAND, fmt.escape_typst(title.upper())),
        "  #v(2.5pt)",
        "  #set text(size: %s)" % tokens.SIZE_PANEL,
        "  #table(columns: (1fr, auto), align: (left, right), stroke: none,",
        "    inset: (x: 0pt, y: 1.7pt),",
    ] + rows + ["  )", "]"]) + "\n"


def _annual_window(rows: Sequence) -> List:
    """Returns the fiscal years a table shows, plus any trailing TTM row.

    The TTM row is held aside and re-appended rather than counted against
    the window, so a table always covers the same FISCAL YEARS as the chart
    beside it. Counting TTM as one of the slots shifted the table's start
    year one later than the chart's, which read as a mismatch.
    """
    rows = list(rows)
    if not rows:
        return []
    trailing: List = []
    if str(getattr(rows[-1], "period", "")).strip().upper() == "TTM":
        trailing = [rows[-1]]
        rows = rows[:-1]
    return rows[-TABLE_PERIODS:] + trailing


# --- cover ---------------------------------------------------------------


def cover(snap: CompanySnapshot, derived: DerivedAnalytics,
          comp: Composites, as_of: str) -> str:
    """Builds the masthead and key-data tearsheet."""
    ratios = snap.key_ratios
    latest_bal = snap.latest_balance
    ttm = snap.ttm
    enterprise = derived.enterprise

    market = [
        ["Implied price (Rs)", fmt.per_share(snap.implied_price)],
        ["Market cap (Rs cr)", fmt.num(snap.market_cap_cr)],
        ["Enterprise value (Rs cr)", fmt.num(enterprise.enterprise_value)],
        ["Shares outstanding (cr)", fmt.num(snap.shares_cr, 1)],
        ["Analyst coverage", "%s recos" % (snap.total_reco if snap.total_reco else fmt.DASH)],
        ["Buy recommendations", fmt.pct(snap.buy_reco_pct, 0)],
    ]
    valuation = [
        ["Trailing P/E (x)", fmt.ratio(ratios.get("ttmPe"), 2)],
        ["Industry P/E (x)", fmt.ratio(ratios.get("indpe"), 2)],
        ["Premium / discount", fmt.signed_pct(snap.pe_discount_to_industry)],
        ["EV / EBITDA (x)", fmt.ratio(enterprise.ev_to_ebitda, 2)],
        ["Price / book (x)", fmt.ratio(ratios.get("pbr"), 2)],
        ["Dividend yield", fmt.pct(ratios.get("divYield"), 2)],
    ]
    performance = [
        ["TTM revenue (Rs cr)", fmt.num(ttm.revenue if ttm else None)],
        ["TTM EBITDA margin", fmt.pct(ttm.ebitda_margin if ttm else None)],
        ["TTM PAT (Rs cr)", fmt.num(ttm.pat if ttm else None)],
        ["TTM EPS (Rs)", fmt.per_share(ttm.eps if ttm else None)],
        ["Return on equity", fmt.pct(snap.dupont.get("return_on_equity_roe_pct"))],
        ["Return on invested capital", fmt.pct(
            snap.capital_efficiency.get("return_on_invested_capital_roic_pct"))],
    ]

    description = snap.description.strip()
    if len(description) > 560:
        description = description[:560].rsplit(" ", 1)[0] + "..."

    out = [
        "#block(below: 6pt)[",
        "  #text(font: %s, size: %s, weight: \"bold\")[%s]" % (
            _font_list(tokens.FONT_DISPLAY), tokens.SIZE_MASTHEAD,
            fmt.escape_typst(snap.name or snap.ticker)),
        "  #v(-4pt)",
        "  #text(size: %s, fill: %s)[%s #sym.dot.c %s #sym.dot.c Fundamental profile, %s]" % (
            tokens.SIZE_SUBTITLE, C_MUTED, fmt.escape_typst(snap.ticker),
            fmt.escape_typst(snap.sector or "Sector not classified"),
            fmt.escape_typst(as_of)),
        "]",
        "#grid(columns: (1fr, 1fr, 1fr), gutter: %s, align: top," % tokens.GUTTER,
        "  [%s]," % _panel("Market data", market),
        "  [%s]," % _panel("Valuation", valuation),
        "  [%s]," % _panel("Trailing performance", performance),
        ")",
        "#v(5pt)",
    ]

    if description:
        out += [
            "#block(inset: (left: 6pt), stroke: (left: 1.6pt + %s))[" % C_BRAND,
            "  #text(size: %s, fill: %s)[%s]" % (
                tokens.SIZE_BODY, C_INK_SOFT, fmt.escape_typst(description)),
            "]",
        ]

    caveats = list(snap.warnings) + list(derived.notes) + list(comp.notes)
    if caveats:
        items = " ".join(fmt.escape_typst(c) for c in caveats)
        out += [
            "#v(4pt)",
            "#block(fill: %s, inset: (x: 6pt, y: 4.5pt), radius: 1.5pt, width: 100%%)[" % C_BRAND_TINT,
            "  #text(size: %s, fill: %s, weight: \"semibold\", tracking: 0.3pt)[DATA LIMITATIONS]" % (
                tokens.SIZE_FOOTNOTE, C_BRAND),
            "  #h(4pt) #text(size: %s, fill: %s)[%s]" % (
                tokens.SIZE_FOOTNOTE, C_INK_SOFT, items),
            "]",
        ]

    return "\n".join(out) + "\n"


# --- section 1: earnings power -------------------------------------------


def earnings_power(
    snap: CompanySnapshot, derived: DerivedAnalytics, comp: Composites,
    check: SelfCheck, ex: Exhibits, charts: Dict[str, Path],
) -> str:
    """Annual income statement and the cost structure that produces it."""
    periods = _annual_window(list(snap.years) + ([snap.ttm] if snap.ttm else []))
    rows: List[List[str]] = []
    for index, row in enumerate(periods):
        prior = periods[index - 1] if index > 0 else None
        rows.append([
            _cell(fmt.period_label(row.period)),
            _cell(fmt.num(row.revenue)),
            _delta_cell(fmt.growth(row.revenue, prior.revenue if prior else None)),
            _cell(fmt.num(row.ebitda)),
            _cell(fmt.num(row.ebit)),
            _cell(fmt.num(row.pat)),
            _delta_cell(fmt.growth(row.pat, prior.pat if prior else None)),
            _cell(fmt.per_share(row.eps)),
        ])

    cost_rows: List[List[str]] = []
    for metrics in _annual_window(derived.annual):
        cost_rows.append([
            _cell(fmt.period_label(metrics.period)),
            _cell(fmt.pct(metrics.raw_material_ratio)),
            _cell(fmt.pct(metrics.sga_ratio)),
            _cell(fmt.pct(metrics.depreciation_ratio)),
            _cell(fmt.pct(metrics.ebitda_margin)),
            _cell(fmt.pct(metrics.ebit_margin)),
            _cell(fmt.pct(metrics.pat_margin)),
            _cell(fmt.pct(metrics.effective_tax_rate)),
        ])

    out = [_section(
        "Earnings power",
        "How much the business earns, and what the cost structure behind it looks like.",
    )]
    out.append(ex.wrap(
        "Annual income statement (Rs cr, EPS in Rs)",
        _table(
            columns="(auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr)",
            align="(left, right, right, right, right, right, right, right)",
            header=["Period", "Revenue", "YoY", "EBITDA", "EBIT", "PAT", "PAT YoY", "EPS"],
            rows=rows,
            emphasise_last=bool(snap.ttm),
        ),
        "Financial Data Collector; EBITDA, growth and margins computed by GrowNXT. "
        "TTM is trailing twelve months, not a fiscal year.",
    ))
    if charts.get("annual") is not None:
        out.append(ex.wrap("Revenue and EBIT with operating margin", _image(charts["annual"])))

    out.append(_two_up(
        ex.wrap("Cost structure and margins (% of revenue)", _table(
            columns="(auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr)",
            align="(left, right, right, right, right, right, right, right)",
            header=["Period", "Raw mat", "SG&A", "D&A", "EBITDA", "EBIT", "PAT", "Tax rate"],
            rows=cost_rows,
            size=tokens.SIZE_SMALL,
        ), "Raw materials is the provider's reported material cost, not total "
            "cost of goods sold; it is immaterial for services businesses."),
        (ex.wrap("Margin trajectory", _image(charts["margins"])))
        if charts.get("margins") is not None else "",
        ratio="(1.55fr, 1fr)",
    ))
    return "".join(out)


# --- section 2: near-term trajectory -------------------------------------


def near_term(
    snap: CompanySnapshot, derived: DerivedAnalytics, comp: Composites,
    check: SelfCheck, ex: Exhibits, charts: Dict[str, Path],
) -> str:
    """Quarterly results and the trailing trend built from them."""
    quarters = snap.quarters[-tokens.DISPLAY_QUARTERS:]
    rows: List[List[str]] = []
    for index, row in enumerate(quarters):
        prior = quarters[index - 1] if index > 0 else None
        year_ago = quarters[index - 4] if index >= 4 else None
        rows.append([
            _cell(fmt.period_label(row.period)),
            _cell(fmt.num(row.revenue)),
            _delta_cell(fmt.growth(row.revenue, prior.revenue if prior else None)),
            _delta_cell(fmt.growth(row.revenue, year_ago.revenue if year_ago else None)),
            _cell(fmt.num(row.ebit)),
            _cell(fmt.pct(row.ebit_margin)),
            _cell(fmt.num(row.pat)),
            _cell(fmt.pct(row.pat_margin)),
            _cell(fmt.per_share(row.eps)),
        ])

    out = [_section(
        "Near-term trajectory",
        "Where momentum sits now, with seasonality removed by a trailing window.",
    )]
    out.append(ex.wrap(
        "Quarterly income statement (Rs cr, EPS in Rs)",
        _table(
            columns="(auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr)",
            align="(left, right, right, right, right, right, right, right, right)",
            header=["Quarter", "Revenue", "QoQ", "YoY", "EBIT", "EBIT %",
                    "PAT", "PAT %", "EPS"],
            rows=rows,
        ),
        "Financial Data Collector. YoY compares the same quarter a year earlier.",
    ))

    left = charts.get("quarterly")
    right = charts.get("rolling")
    if left is not None and right is not None:
        out.append(_two_up(
            ex.wrap("Quarterly revenue and PAT margin", _image(left)),
            ex.wrap("Trailing twelve-month revenue and margin", _image(right), "Financial Data Collector; four-quarter rolling sums by GrowNXT"),
        ))
    elif left is not None:
        out.append(ex.wrap("Quarterly revenue and PAT margin", _image(left)))
    return "".join(out)


# --- section 3: returns on capital ---------------------------------------


def returns_on_capital(
    snap: CompanySnapshot, derived: DerivedAnalytics, comp: Composites,
    check: SelfCheck, ex: Exhibits, charts: Dict[str, Path],
) -> str:
    """Return series, plus the DuPont decomposition across every period."""
    rows: List[List[str]] = []
    for metrics in _annual_window(derived.annual):
        rows.append([
            _cell(fmt.period_label(metrics.period)),
            _cell(fmt.pct(metrics.roe)),
            _cell(fmt.pct(metrics.roce)),
            _cell(fmt.pct(metrics.roic)),
            _cell(fmt.num(metrics.nopat)),
            _cell(fmt.num(metrics.invested_capital)),
            _cell(fmt.pct(metrics.effective_tax_rate)),
        ])

    dupont = snap.dupont
    dupont_rows = [
        [_cell("Tax burden"), _cell(fmt.ratio(dupont.get("tax_burden_ratio"), 4)),
         _cell("PAT / PBT")],
        [_cell("Interest burden"), _cell(fmt.ratio(dupont.get("interest_burden_ratio"), 4)),
         _cell("PBT / EBIT")],
        [_cell("Operating margin"), _cell(fmt.pct(dupont.get("operating_margin_pct"), 2)),
         _cell("EBIT / revenue")],
        [_cell("Asset turnover"), _cell(fmt.mult(dupont.get("asset_turnover_x"))),
         _cell("Revenue / assets")],
        [_cell("Equity multiplier"), _cell(fmt.mult(dupont.get("equity_multiplier_x"))),
         _cell("Assets / equity")],
        [_bold_cell("Return on equity"),
         _bold_cell(fmt.pct(dupont.get("return_on_equity_roe_pct"), 2)),
         _cell("Product of the five factors")],
    ]

    out = [_section(
        "Returns on capital",
        "What the business earns on the money tied up in it, and why that has moved.",
    )]
    out.append(ex.wrap(
        "Return series (Rs cr where absolute)",
        _table(
            columns="(auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr)",
            align="(left, right, right, right, right, right, right)",
            header=["Period", "ROE", "ROCE", "ROIC", "NOPAT", "Invested capital", "Tax rate"],
            rows=rows,
            emphasise_last=bool(snap.ttm),
        ),
        "Computed by GrowNXT on period-end capital, matching the provider's own "
        "DuPont and capital-efficiency methodology.",
    ))
    out.append(_two_up(
        ex.wrap(
            "DuPont decomposition as published by the provider (%s)"
            % str(dupont.get("period") or "TTM"),
            _table(
                columns="(auto, auto, 1fr)",
                align="(left, right, left)",
                header=["Factor", "Value", "Definition"],
                rows=dupont_rows,
            ),
        ),
        (ex.wrap("Return trend", _image(charts["returns"])))
        if charts.get("returns") is not None else "",
        ratio="(1.2fr, 1fr)",
    ))
    out.append(_dupont_series(comp, ex, charts))
    return "".join(out)


def _dupont_series(comp: Composites, ex: Exhibits, charts: Dict[str, Path]) -> str:
    """The five-factor decomposition extended across every reported period.

    The provider publishes one trailing column. A single column locates
    returns; the series is what identifies the factor that moved them, which
    is the only reason to decompose anything. The final two columns are the
    exhibit auditing itself: ROE rebuilt from the row's own five factors,
    against ROE taken straight from profit over equity.
    """
    series = comp.dupont
    if not series.years:
        return ""

    rows: List[List[str]] = []
    for row in _annual_window(series.years):
        rows.append([
            _cell(fmt.period_label(row.period)),
            _cell(fmt.ratio(row.tax_burden, 4)),
            _cell(fmt.ratio(row.interest_burden, 4)),
            _cell(fmt.pct(row.operating_margin, 2)),
            _cell(fmt.mult(row.asset_turnover)),
            _cell(fmt.mult(row.equity_multiplier)),
            _cell(fmt.pct(row.roe_product, 2)),
            _cell(fmt.pct(row.roe_direct, 2)),
            _residual_cell(row.residual),
        ])

    note = (
        "Computed by GrowNXT from the reported statements. Tax burden is PAT "
        "over PBT, interest burden PBT over EBIT, and the five factors "
        "multiply to return on equity. The residual column is the row checked "
        "against itself: ROE rebuilt from the factors, less ROE taken directly "
        "from profit over period-end equity."
    )
    if series.reconciliation_delta is not None:
        note += (" The %s column reconciles to the provider's own five-factor "
                 "endpoint within %s." % (
                     fmt.period_label(series.provider_period),
                     fmt.pct(abs(series.reconciliation_delta), 4)))

    table = ex.wrap(
        "DuPont decomposition, every reported period",
        _table(
            columns="(auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, auto)",
            align="(left, right, right, right, right, right, right, right, right)",
            header=["Period", "Tax burden", "Interest burden", "Operating margin",
                    "Asset turnover", "Equity multiplier", "ROE rebuilt",
                    "ROE direct", "Residual"],
            rows=rows,
            size=tokens.SIZE_SMALL,
            emphasise_last=any(r.period.strip().upper() == "TTM"
                               for r in series.years),
        ),
        note,
    )
    if charts.get("dupont") is None:
        return table
    return table + ex.wrap(
        "Which factor moved returns",
        _image(charts["dupont"], "64%"),
        "Computed by GrowNXT. Each factor is rebased to 100 at the first year "
        "shown, because the five sit at incompatible natural scales: a burden "
        "ratio near 0.8 and an operating margin near 22 cannot share an axis "
        "without hiding one of them. Levels are in the table above.",
    )


# --- section 4: financial position --------------------------------------


def financial_position(
    snap: CompanySnapshot, derived: DerivedAnalytics, comp: Composites,
    check: SelfCheck, ex: Exhibits, charts: Dict[str, Path],
) -> str:
    """Balance sheet with the leverage and coverage series drawn from it."""
    balance_rows = [[
        _cell(fmt.period_label(row.period)),
        _cell(fmt.num(row.equity)),
        _cell(fmt.num(row.debt)),
        _cell(fmt.num(row.cash)),
        _cell(fmt.num(row.net_debt)),
        _cell(fmt.num(row.total_assets)),
        _cell(fmt.per_share(row.book_value_per_share)),
    ] for row in _annual_window(snap.balance)]

    # Only periods with at least one computable leverage measure earn a row;
    # when none do, the exhibit is dropped entirely rather than printed as a
    # grid of em-dashes, and the reason appears in the limitations band.
    leverage_source = [m for m in _annual_window(derived.annual)
                       if any(v is not None for v in (
                           m.debt_to_equity, m.net_debt_to_ebitda,
                           m.interest_coverage, m.short_term_debt_share,
                           m.goodwill_to_equity))]
    leverage_rows = [[
        _cell(fmt.period_label(m.period)),
        _cell(fmt.mult(m.debt_to_equity)),
        _cell(fmt.mult(m.net_debt_to_ebitda)),
        _cell(fmt.mult(m.interest_coverage)),
        _cell(fmt.pct(m.short_term_debt_share)),
        _cell(fmt.pct(m.goodwill_to_equity)),
    ] for m in leverage_source]

    out = [_section(
        "Financial position",
        "How the balance sheet is funded, and how much room it leaves.",
    )]
    out.append(ex.wrap(
        "Balance sheet (Rs cr, BV/share in Rs)",
        _table(
            columns="(auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr)",
            align="(left, right, right, right, right, right, right)",
            header=["Period", "Equity", "Debt", "Cash", "Net debt", "Assets", "BV/sh"],
            rows=balance_rows,
        ),
        "Financial Data Collector. Negative net debt indicates a net cash position.",
    ))
    leverage_exhibit = ex.wrap(
        "Leverage and coverage",
        _table(
            columns="(auto, 1fr, 1fr, 1fr, 1fr, 1fr)",
            align="(left, right, right, right, right, right)",
            header=["Period", "D/E", "Net debt / EBITDA", "Interest cover",
                    "Short-term debt", "Goodwill / equity"],
            rows=leverage_rows,
            size=tokens.SIZE_SMALL,
        ),
        "Ratios with a negative EBITDA or interest base are withheld rather "
        "than shown with a misleading sign.",
    ) if leverage_rows else ""

    has_capital_chart = charts.get("capital") is not None
    if leverage_exhibit and has_capital_chart:
        out.append(_two_up(
            leverage_exhibit,
            ex.wrap("Capital structure", _image(charts["capital"])),
            ratio="(1.35fr, 1fr)",
        ))
    elif has_capital_chart:
        # With no leverage table to sit beside, the chart is held to a
        # contained width rather than stretched across the full measure.
        out.append(ex.wrap("Capital structure", _image(charts["capital"], "64%")))
    elif leverage_exhibit:
        out.append(leverage_exhibit)
    return "".join(out)


# --- section 5: cash and earnings quality --------------------------------


def cash_and_quality(
    snap: CompanySnapshot, derived: DerivedAnalytics, comp: Composites,
    check: SelfCheck, ex: Exhibits, charts: Dict[str, Path],
) -> str:
    """Cash flow, earnings-quality tests and the working-capital cycle.

    Grouped because all three ask the same question from different angles:
    is the reported profit real, and does it arrive as cash?
    """
    cash_rows = [[
        _cell(fmt.period_label(row.period)),
        _cell(fmt.num(row.cfo)),
        _cell(fmt.num(row.capex)),
        _cell(fmt.num(row.fcf)),
        _cell(fmt.num(row.dividends_paid)),
    ] for row in _annual_window(snap.cashflow)]

    # These families are cash-flow and balance-sheet derived, and the
    # provider publishes no TTM cash flow, so a trailing row would be
    # entirely em-dashes. Drop rows with nothing in them rather than
    # printing an empty one.
    quality_source = [m for m in _annual_window(derived.annual)
                      if any(v is not None for v in (
                          m.cfo_to_pat, m.accrual_ratio, m.fcf_margin,
                          m.capex_intensity, m.capex_to_depreciation, m.retained_fcf))]
    quality_rows = [[
        _cell(fmt.period_label(m.period)),
        _cell(fmt.pct(m.cfo_to_pat)),
        _cell(fmt.pct(m.accrual_ratio)),
        _cell(fmt.pct(m.fcf_margin)),
        _cell(fmt.pct(m.capex_intensity)),
        _cell(fmt.mult(m.capex_to_depreciation)),
        _cell(fmt.num(m.retained_fcf)),
    ] for m in quality_source]

    workcap_source = [m for m in _annual_window(derived.annual)
                      if m.cash_conversion_cycle is not None]
    workcap_rows = [[
        _cell(fmt.period_label(m.period)),
        _cell(fmt.days(m.dso)),
        _cell(fmt.days(m.dio)),
        _cell(fmt.days(m.dpo)),
        _cell(fmt.days(m.cash_conversion_cycle)),
    ] for m in workcap_source]

    out = [_section(
        "Cash generation and earnings quality",
        "Whether reported profit converts into cash, and where working capital absorbs it.",
    )]
    out.append(ex.wrap(
        "Earnings quality and reinvestment",
        _table(
            columns="(auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr)",
            align="(left, right, right, right, right, right, right)",
            header=["Period", "CFO / PAT", "Accruals / assets", "FCF margin",
                    "Capex / sales", "Capex / D&A", "FCF after dividends"],
            rows=quality_rows,
            size=tokens.SIZE_SMALL,
        ),
        "Computed by GrowNXT. Accruals are (PAT less CFO) over average total "
        "assets; a persistently positive figure means profit is not arriving as "
        "cash. Capex below depreciation indicates under-investment.",
    ))
    out.append(_two_up(
        ex.wrap("Cash flow statement (Rs cr)", _table(
            columns="(auto, 1fr, 1fr, 1fr, 1fr)",
            align="(left, right, right, right, right)",
            header=["Period", "Operating CF", "Capex", "Free CF", "Dividends"],
            rows=cash_rows,
        ), "Financial Data Collector"),
        (ex.wrap("Cash generation and reinvestment", _image(charts["cash"])))
        if charts.get("cash") is not None else "",
        ratio="(1.1fr, 1fr)",
    ))

    if any(m.cash_conversion_cycle is not None for m in derived.annual):
        out.append(_two_up(
            ex.wrap("Working-capital cycle (days)", _table(
                columns="(auto, 1fr, 1fr, 1fr, 1fr)",
                align="(left, right, right, right, right)",
                header=["Period", "DSO", "DIO", "DPO", "Cash cycle"],
                rows=workcap_rows,
            ), "Computed by GrowNXT. A negative cycle means suppliers fund "
                "operations. Inventory and payable days use reported material "
                "cost as the base."),
            (ex.wrap("Working-capital days", _image(charts["workcap"])))
            if charts.get("workcap") is not None else "",
            ratio="(1fr, 1.15fr)",
        ))
    return "".join(out)


# --- section 6: capital allocation ---------------------------------------


def capital_allocation(
    snap: CompanySnapshot, derived: DerivedAnalytics, comp: Composites,
    check: SelfCheck, ex: Exhibits, charts: Dict[str, Path],
) -> str:
    """Where the cash went, and whether what was put back in earned its keep.

    These two exhibits belong together and nowhere else. The first traces
    every rupee of cash to a destination; the second asks whether the portion
    that went back into the business bought the growth that followed. Cash
    generation, in the section before, establishes how much there was to
    allocate; valuation, in the section after, prices the result.
    """
    allocation = comp.sources_uses
    out = [_section(
        "Capital allocation",
        "Where the cash generated has gone, and what the reinvested share bought.",
    )]

    if not allocation.sources and not allocation.uses:
        for note in allocation.notes:
            out.append(_note_block(note))
        return "".join(out)

    source_rows = [[
        _cell(fmt.escape_typst(item.label)),
        _cell(fmt.num(item.magnitude)),
        _cell(fmt.pct(allocation.share(item, False))),
    ] for item in allocation.sources]
    source_rows.append([
        _bold_cell("Total sources"),
        _bold_cell(fmt.num(allocation.total_sources)),
        _bold_cell(fmt.pct(100.0)),
    ])

    use_rows = [[
        _cell(fmt.escape_typst(item.label)),
        _cell(fmt.num(item.magnitude)),
        _cell(fmt.pct(allocation.share(item, True))),
    ] for item in allocation.uses]
    use_rows.append([
        _bold_cell("Total uses"),
        _bold_cell(fmt.num(allocation.total_uses)),
        _bold_cell(fmt.pct(100.0)),
    ])

    window = "%s to %s" % (fmt.period_label(allocation.from_period),
                           fmt.period_label(allocation.to_period))
    residual_note = (
        "Built on the cash flow statement's own articulation, so the two "
        "columns agree by construction rather than by assertion; the "
        "verification section reports the residual. Two lines are residuals "
        "rather than reported figures, because the provider publishes net "
        "investing and net financing totals but no debt-raised, debt-repaid "
        "or buyback line."
    )
    if allocation.debt_change is not None:
        direction = "rose" if allocation.debt_change >= 0 else "fell"
        residual_note += (
            " Total debt %s by %s Rs cr over the window, which is what sits "
            "inside the net financing line alongside any buyback."
            % (direction, fmt.num(abs(allocation.debt_change))))
    if allocation.share_change_pct is not None:
        residual_note += (" The share count moved %s over the same period."
                          % fmt.signed_pct(allocation.share_change_pct, 2))

    out.append(_two_up(
        ex.wrap("Sources of cash, %s (Rs cr)" % window, _table(
            columns="(1fr, auto, auto)",
            align="(left, right, right)",
            header=["Source", "Amount", "Share"],
            rows=source_rows,
            emphasise_last=True,
        ), "Cumulative over %d years. Computed by GrowNXT." % allocation.years),
        ex.wrap("Uses of cash, %s (Rs cr)" % window, _table(
            columns="(1fr, auto, auto)",
            align="(left, right, right)",
            header=["Use", "Amount", "Share"],
            rows=use_rows,
            emphasise_last=True,
        ), "Cumulative over %d years. Computed by GrowNXT." % allocation.years),
        ratio="(1fr, 1fr)",
    ))
    # Full width beneath the pair rather than hung off one table's source
    # line: it explains both columns, and a long note under the shorter of
    # two side-by-side tables unbalances them.
    out.append(_note_block(residual_note))
    if charts.get("allocation") is not None:
        out.append(ex.wrap(
            "Sources and uses as matched compositions",
            _image(charts["allocation"]),
            "Computed by GrowNXT. The two bars are the same length because "
            "sources equal uses; the exhibit shows the mix, and the tables "
            "above carry the amounts. Segments run in the same order as "
            "those tables, and a segment too narrow to hold its name "
            "carries its share alone.",
        ))
    for note in allocation.notes:
        out.append(_note_block(note))

    out.append(_reinvestment_exhibits(comp, ex, charts))
    return "".join(out)


def _reinvestment_exhibits(
    comp: Composites, ex: Exhibits, charts: Dict[str, Path],
) -> str:
    """The reinvestment identity, per year and in aggregate."""
    reinvestment = comp.reinvestment
    if reinvestment.withheld_reason:
        return _note_block(reinvestment.withheld_reason)
    if not reinvestment.years:
        return ""

    rows = [[
        _cell(fmt.period_label(year.period)),
        _cell(fmt.num(year.nopat)),
        _cell(fmt.num(year.capex)),
        _cell(fmt.num(year.depreciation)),
        _cell(fmt.num(year.net_capex)),
        _cell(fmt.num(year.delta_working_capital)),
        _cell(fmt.num(year.reinvestment)),
        _cell(fmt.pct(year.reinvestment_rate)),
        _cell(fmt.pct(year.roic)),
        _delta_cell(year.implied_growth),
        _delta_cell(year.revenue_growth),
    ] for year in reinvestment.years]

    summary = [
        ["Reinvestment over the window (Rs cr)", fmt.num(reinvestment.total_reinvestment)],
        ["NOPAT over the window (Rs cr)", fmt.num(reinvestment.total_nopat)],
        ["Reinvestment rate", fmt.pct(reinvestment.aggregate_reinvestment_rate)],
        ["Average return on invested capital", fmt.pct(reinvestment.average_roic)],
        ["Implied growth rate", fmt.signed_pct(reinvestment.implied_growth, 2)],
        ["Revenue growth delivered", fmt.signed_pct(reinvestment.actual_revenue_cagr, 2)],
        ["EBIT growth delivered", fmt.signed_pct(reinvestment.actual_ebit_cagr, 2)],
        ["Unexplained gap", fmt.signed_pct(reinvestment.growth_gap, 2)],
    ]

    out = [ex.wrap(
        "The reinvestment identity, year by year (Rs cr unless marked)",
        _table(
            columns="(auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr)",
            align="(left, right, right, right, right, right, right, right, "
                  "right, right, right)",
            header=["Period", "NOPAT", "Capex", "D&A", "Net capex",
                    "WC change", "Reinvested", "Rate", "ROIC",
                    "Implied g", "Actual g"],
            rows=rows,
            size=tokens.SIZE_SMALL,
        ),
        "Computed by GrowNXT. Reinvestment is capital expenditure less "
        "depreciation, plus the increase in non-cash working capital, which "
        "treats depreciation as maintenance capital. Working capital excludes "
        "cash and short-term debt because both are financing rather than "
        "operating decisions. The rate is withheld where NOPAT is not "
        "positive.",
    )]
    out.append(_two_up(
        ex.wrap("Growth the identity implies, against growth delivered", _panel(
            "%s to %s" % (fmt.period_label(reinvestment.window_from),
                          fmt.period_label(reinvestment.window_to)),
            summary,
        ), "Computed by GrowNXT. The rate is aggregated over the window rather "
           "than averaged across years, so one restructuring year cannot "
           "dominate the answer."),
        (ex.wrap("Implied against delivered growth", _image(charts["reinvestment"])))
        if charts.get("reinvestment") is not None else "",
        ratio="(1fr, 1.15fr)",
    ))
    for note in reinvestment.notes:
        out.append(_note_block(note))
    return "".join(out)


# --- section 7: composite scores ------------------------------------------


def composite_scores(
    snap: CompanySnapshot, derived: DerivedAnalytics, comp: Composites,
    check: SelfCheck, ex: Exhibits, charts: Dict[str, Path],
) -> str:
    """The two published scoring frameworks, shown as their sub-tests.

    Neither score is presented as a number with a verdict attached. The
    F-Score exhibit is the nine signals, and the total is the column summed;
    the Z-Score exhibit is the five weighted terms, and the total is those
    added up. A reader who disagrees with a component can see exactly which
    one and recompute without the score.
    """
    out = [_section(
        "Composite quality and solvency scores",
        "What two standard frameworks conclude, and the evidence each one rests on.",
    )]
    out.append(_piotroski_exhibits(comp, ex, charts))
    out.append(_altman_exhibits(comp, ex, charts))
    return "".join(out)


def _piotroski_exhibits(
    comp: Composites, ex: Exhibits, charts: Dict[str, Path],
) -> str:
    """The nine-signal F-Score, its sub-tests and its history."""
    score = comp.piotroski
    if not score.tests:
        return "".join(_note_block(reason) for reason in score.withheld)

    rows = [[
        _cell(str(test.number)),
        _cell(fmt.escape_typst(test.name)),
        _cell(fmt.escape_typst(test.definition)),
        _cell(_score_value(test.value, test.unit)),
        _cell(_score_value(test.comparator, test.unit)),
        _verdict_cell(test.passed, test.points),
    ] for test in score.tests]

    # The score goes in the point column, expressed against the number of
    # signals evaluated. A bare numerator in a column of ones and zeros
    # invites the reader to take it as a ninth row rather than as the total,
    # and the denominator is the half that makes it meaningful.
    rows.append([
        _bold_cell(""),
        _bold_cell("Total, %s" % fmt.period_label(score.period)),
        _bold_cell("Sum of the point column"),
        _bold_cell(""),
        _bold_cell(""),
        _bold_cell(_group_label(score.score, score.computable)),
    ])

    groups = [
        ["Profitability (signals 1-4)", _group_label(score.profitability_points, 4)],
        ["Leverage and liquidity (5-7)", _group_label(score.leverage_points, 3)],
        ["Operating efficiency (8-9)", _group_label(score.efficiency_points, 2)],
        ["Signals evaluated", "%d of 9" % score.computable],
        ["Score", _group_label(score.score, score.computable)],
        ["Compared with", fmt.period_label(score.prior_period)],
    ]

    out = [ex.wrap(
        "Piotroski F-Score, signal by signal (%s against %s)" % (
            fmt.period_label(score.period), fmt.period_label(score.prior_period)),
        _table(
            columns="(auto, 1.5fr, 2.6fr, auto, auto, auto)",
            align="(right, left, left, right, right, center)",
            header=["No.", "Signal", "Test applied", "Value",
                    "Compared with", "Point"],
            rows=rows,
            size=tokens.SIZE_SMALL,
            emphasise_last=True,
        ),
        "Computed by GrowNXT from the reported statements, following Piotroski "
        "(2000). One point per signal passed; the total is the point column "
        "added up, and is shown only alongside the signals that produced it.",
    )]

    # The caveats sit in the left column beneath the summary panel rather
    # than full width below the pair. A six-row panel is a good deal shorter
    # than the chart beside it, and the notes are what that column is for.
    left = [ex.wrap("Score by signal group", _panel("F-SCORE SUMMARY", groups),
                    "Piotroski reads eight or nine as financially "
                    "strengthening and nil or one as deteriorating.")]
    for reason in score.withheld:
        left.append(_note_block(reason, tone="alert"))
    if not score.comparable and score.tests:
        left.append(_note_block(
            "This total is out of %d evaluated signals, not nine, and is "
            "therefore not comparable with a standard F-Score quoted "
            "elsewhere." % score.computable, tone="alert"))
    for substitution in score.substitutions:
        left.append(_note_block(substitution))

    out.append(_two_up(
        "".join(left),
        (ex.wrap("F-Score history", _image(charts["fscore"]),
                 "Computed by GrowNXT. The dashed ceiling is the number of "
                 "signals that could be evaluated, which is what the score "
                 "must be read against."))
        if charts.get("fscore") is not None else "",
        ratio="(1.15fr, 1fr)",
    ))
    return "".join(out)


def _group_label(points: Optional[int], out_of: int) -> str:
    """Renders a sub-score as a fraction, never as a bare numerator."""
    if points is None:
        return fmt.DASH
    return "%d of %d" % (points, out_of)


def _altman_exhibits(
    comp: Composites, ex: Exhibits, charts: Dict[str, Path],
) -> str:
    """The Z-Score, its five weighted terms and the Z-prime history."""
    altman = comp.altman
    if altman.withheld_reason:
        note = _note_block(altman.withheld_reason, tone="alert")
        return note

    rows = [[
        _cell(fmt.escape_typst(component.name)),
        _cell(fmt.escape_typst(component.definition)),
        _cell(fmt.ratio(component.ratio, 4)),
        _cell(fmt.ratio(component.weight, 3)),
        _cell(fmt.ratio(component.contribution, 4)),
    ] for component in altman.components]
    rows.append([
        _bold_cell("Z-Score"),
        _bold_cell("Sum of the weighted terms above"),
        _bold_cell(""),
        _bold_cell(""),
        _bold_cell(fmt.ratio(altman.score, 3)),
    ])

    bands = [
        ["Safe zone", "above %s" % fmt.ratio(altman.safe_above)],
        ["Grey zone", "%s to %s" % (fmt.ratio(altman.distress_below),
                                    fmt.ratio(altman.safe_above))],
        ["Distress zone", "below %s" % fmt.ratio(altman.distress_below)],
        ["Score", fmt.ratio(altman.score, 3)],
        ["Zone", altman.zone or fmt.DASH],
        ["Equity term taken at", altman.equity_basis or fmt.DASH],
    ]

    out = [ex.wrap(
        "Altman Z-Score, term by term (%s)" % fmt.period_label(altman.period),
        _table(
            columns="(1.3fr, 2.2fr, auto, auto, auto)",
            align="(left, left, right, right, right)",
            header=["Term", "Definition", "Ratio", "Coefficient",
                    "Contribution"],
            rows=rows,
            size=tokens.SIZE_SMALL,
            emphasise_last=True,
        ),
        "Computed by GrowNXT following Altman (1968). Coefficients are the "
        "published ones and are not fitted here. The score is the contribution "
        "column added up, and is shown only alongside the terms that produced "
        "it.",
    )]

    left = [ex.wrap("Zones and where this score falls",
                    _panel("Z-SCORE READING", bands),
                    "Thresholds are Altman's published cut-offs for the "
                    "listed-company model.")]

    dominant = altman.dominant
    if dominant is not None:
        left.append(_note_block(
            "One term supplies %s of this score: %s, at %s of the %s total. "
            "Altman fitted the model on manufacturers, and that term "
            "misbehaves for an asset-light company whose market value is large "
            "against a small liability base, where it reports a valuation "
            "observation rather than a solvency one. Read the term, not the "
            "total." % (
                fmt.pct(altman.concentration_pct, 0),
                dominant.name.lower(),
                fmt.ratio(dominant.contribution, 3),
                fmt.ratio(altman.score, 3),
            ), tone="alert"))
    if altman.equity_basis.startswith("Implied"):
        left.append(_note_block(
            "The equity term uses a market capitalisation inferred from the "
            "trailing price-to-earnings ratio, because the provider exposes no "
            "live quote. It is a derivation rather than a market price, and the "
            "score moves with it."))

    out.append(_two_up(
        "".join(left),
        (ex.wrap("Z-prime history against the distress bands",
                 _image(charts["altman"]),
                 "Computed by GrowNXT. The history uses Altman's 1983 "
                 "private-company revision, because no share price is "
                 "available for a past year and the equity term must be taken "
                 "at book. Its coefficients and cut-offs differ from the "
                 "listed model above, so read the direction of this series "
                 "rather than comparing its level with the score beside it."))
        if charts.get("altman") is not None else "",
        ratio="(1.15fr, 1fr)",
    ))
    return "".join(out)


# --- section 8: valuation -----------------------------------------------


def valuation(
    snap: CompanySnapshot, derived: DerivedAnalytics, comp: Composites,
    check: SelfCheck, ex: Exhibits, charts: Dict[str, Path],
) -> str:
    """Enterprise-value bridge, multiples, and the peer set."""
    enterprise = derived.enterprise
    ratios = snap.key_ratios

    bridge_rows = [
        [_cell("Market capitalisation"), _cell(fmt.num(enterprise.market_cap))],
        [_cell("Add: total debt"), _cell(fmt.num(enterprise.total_debt))],
        [_cell("Less: cash and equivalents"),
         _cell(fmt.num(-enterprise.cash if enterprise.cash is not None else None))],
        [_cell("Add: minority interest"), _cell(fmt.num(enterprise.minority_interest))],
        [_bold_cell("Enterprise value"), _bold_cell(fmt.num(enterprise.enterprise_value))],
    ]

    multiple_rows = [
        [_cell("Trailing P/E (x)"), _cell(fmt.ratio(ratios.get("ttmPe"), 2)),
         _cell(fmt.ratio(ratios.get("indpe"), 2))],
        [_cell("Price / book (x)"), _cell(fmt.ratio(ratios.get("pbr"), 2)),
         _cell(fmt.ratio(ratios.get("indpb"), 2))],
        [_cell("Dividend yield"), _cell(fmt.pct(ratios.get("divYield"), 2)),
         _cell(fmt.pct(ratios.get("inddy"), 2))],
        [_cell("EV / EBITDA (x)"), _cell(fmt.ratio(enterprise.ev_to_ebitda, 2)),
         _cell(fmt.DASH)],
        [_cell("EV / EBIT (x)"), _cell(fmt.ratio(enterprise.ev_to_ebit, 2)),
         _cell(fmt.DASH)],
        [_cell("EV / sales (x)"), _cell(fmt.ratio(enterprise.ev_to_sales, 2)),
         _cell(fmt.DASH)],
        [_cell("EV / free cash flow (x)"), _cell(fmt.ratio(enterprise.ev_to_fcf, 2)),
         _cell(fmt.DASH)],
        [_cell("Free cash flow yield"), _cell(fmt.pct(enterprise.fcf_yield, 2)),
         _cell(fmt.DASH)],
        [_cell("Earnings yield"), _cell(fmt.pct(enterprise.earnings_yield, 2)),
         _cell(fmt.DASH)],
    ]

    peer_rows: List[List[str]] = []
    for peer in snap.peers:
        name = fmt.escape_typst(peer.name or peer.ticker)
        label = ("#text(weight: \"semibold\", fill: %s)[%s]" % (C_BRAND, name)
                 if peer.is_subject else name)
        peer_rows.append([
            _cell(label),
            _cell(fmt.escape_typst(peer.ticker)),
            _cell(fmt.num(peer.market_cap_cr)),
            _cell(fmt.ratio(peer.pe, 2)),
            _cell(fmt.ratio(peer.pb, 2)),
            _cell(fmt.pct(peer.div_yield, 2)),
            _delta_cell(peer.change_52w),
            _cell(fmt.pct(peer.buy_reco_pct, 0)),
        ])

    out = [_section(
        "Valuation",
        "What the market is paying, on equity and on the whole enterprise.",
    )]
    out.append(_two_up(
        ex.wrap("Enterprise value bridge (Rs cr)", _table(
            columns="(1fr, auto)",
            align="(left, right)",
            header=["Component", "Amount"],
            rows=bridge_rows,
            emphasise_last=True,
        ), "Computed by GrowNXT from the latest balance sheet. Minority "
            "interest is included because enterprise earnings consolidate it."),
        ex.wrap("Valuation multiples", _table(
            columns="(1fr, auto, auto)",
            align="(left, right, right)",
            header=["Measure", "Company", "Industry"],
            rows=multiple_rows,
        ), "P/E, P/B and yield from the provider; enterprise multiples computed "
            "by GrowNXT. No industry benchmark is published for EV measures."),
        ratio="(1fr, 1.05fr)",
    ))
    out.append(ex.wrap(
        "Peer comparison",
        _table(
            columns="(1.7fr, auto, 1fr, auto, auto, auto, auto, auto)",
            align="(left, left, right, right, right, right, right, right)",
            header=["Company", "Ticker", "Mkt cap (Rs cr)", "P/E", "P/B",
                    "Div yld", "52-week", "Buy recos"],
            rows=peer_rows,
        ),
        "Financial Data Collector. Market caps converted from the provider's "
        "rupee-million denomination. Subject company highlighted.",
    ))
    if charts.get("peers") is not None:
        out.append(ex.wrap(
            "Peer multiples ranked against the industry",
            _image(charts["peers"]),
            "Financial Data Collector. Dashed line marks the industry multiple.",
        ))
    return "".join(out)


# --- section 7: shareholder returns and ownership ------------------------


def shareholder_returns(
    snap: CompanySnapshot, derived: DerivedAnalytics, comp: Composites,
    check: SelfCheck, ex: Exhibits, charts: Dict[str, Path],
) -> str:
    """Per-share economics, the EPS bridge, ownership and dividends."""
    per_share_rows = [[
        _cell(fmt.period_label(m.period)),
        _cell(fmt.num(m.shares_cr, 1)),
        _cell(fmt.per_share(m.eps)),
        _cell(fmt.per_share(m.dps)),
        _cell(fmt.pct(m.payout_pct)),
        _cell(fmt.per_share(m.book_value_per_share)),
    ] for m in _annual_window(derived.annual)]

    bridge = derived.per_share
    bridge_rows = [
        [_cell("EPS, %s" % (bridge.from_period or fmt.DASH)),
         _cell(fmt.per_share(bridge.eps_start))],
        [_cell("Change from profit growth"), _cell(fmt.per_share(bridge.profit_effect))],
        [_cell("Change from share count"), _cell(fmt.per_share(bridge.share_count_effect))],
        [_bold_cell("EPS, %s" % (bridge.to_period or fmt.DASH)),
         _bold_cell(fmt.per_share(bridge.eps_end))],
    ]

    holding_rows = [[
        _cell(h.date),
        _cell(fmt.pct(h.promoter, 2)),
        _cell(fmt.pct(h.fii, 2)),
        _cell(fmt.pct(h.dii, 2)),
        _cell(fmt.pct(h.mutual_fund, 2)),
    ] for h in snap.holdings]

    dividend_rows = [[
        _cell(d.ex_date),
        _cell(fmt.escape_typst(d.kind or fmt.DASH)),
        _cell(fmt.per_share(d.amount)),
    ] for d in snap.dividends]

    share_note = "Computed by GrowNXT"
    if bridge.share_change_pct is not None:
        share_note = (
            "Computed by GrowNXT. Share count moved %s over the period, from "
            "%s cr to %s cr." % (
                fmt.signed_pct(bridge.share_change_pct),
                fmt.num(bridge.shares_start, 1), fmt.num(bridge.shares_end, 1))
        )

    out = [_section(
        "Shareholder returns and ownership",
        "What accrues per share, how much is paid out, and who owns the register.",
    )]
    out.append(_two_up(
        ex.wrap("Per-share economics", _table(
            columns="(auto, 1fr, 1fr, 1fr, 1fr, 1fr)",
            align="(left, right, right, right, right, right)",
            header=["Period", "Shares (cr)", "EPS", "DPS", "Payout", "BV/share"],
            rows=per_share_rows,
        ), "Financial Data Collector; payout cross-checked against DPS over EPS."),
        ex.wrap("What moved earnings per share", _table(
            columns="(1fr, auto)",
            align="(left, right)",
            header=["Driver", "Rs per share"],
            rows=bridge_rows,
            emphasise_last=True,
        ), share_note),
        ratio="(1.4fr, 1fr)",
    ))
    holding_table = ex.wrap("Shareholding pattern (%)", _table(
        columns="(auto, 1fr, 1fr, 1fr, 1fr)",
        align="(left, right, right, right, right)",
        header=["Quarter end", "Promoter", "FII", "DII", "MF"],
        rows=holding_rows,
    ), "Financial Data Collector")
    if charts.get("holding") is not None:
        out.append(_two_up(
            holding_table,
            ex.wrap("Institutional holdings trend (%)", _image(charts["holding"])),
            ratio="(1fr, 1.1fr)",
        ))
    else:
        out.append(holding_table)
    out.append(ex.wrap(
        "Declared dividend history",
        _table(
            columns="(auto, auto, 1fr)",
            align="(left, left, right)",
            header=["Ex-date", "Type", "Amount per share (Rs)"],
            rows=dividend_rows,
        ),
        "Financial Data Collector",
    ))
    return "".join(out)


def verification(
    snap: CompanySnapshot, derived: DerivedAnalytics, comp: Composites,
    check: SelfCheck, ex: Exhibits, charts: Dict[str, Path],
) -> str:
    """The report's own arithmetic, re-derived and reported.

    This section exists because a typeset PDF gives a wrong figure the same
    authority as a right one. Every identity the document relies on is
    re-derived by an independent route and the residual printed, so the claim
    that the arithmetic holds is checkable rather than asserted. A failure is
    published in the same table as a pass; suppressing it would defeat the
    purpose of running the check.
    """
    if not check.checks:
        return ""

    rows: List[List[str]] = []
    for item in check.checks:
        if item.passed is None:
            verdict = "[#text(fill: %s)[not applicable]]" % C_MUTED
        elif item.passed:
            verdict = "[#text(fill: %s, weight: \"semibold\")[closes]]" % C_POS
        else:
            verdict = "[#text(fill: %s, weight: \"semibold\")[FAILS]]" % C_NEG
        rows.append([
            _cell(fmt.escape_typst(item.name)),
            _cell(fmt.escape_typst(item.kind)),
            _cell(_verification_number(item.delta, item.unit)),
            _cell(_verification_number(item.tolerance, item.unit)),
            verdict,
        ])

    summary = [
        ["Checks run", str(len(check.applicable))],
        ["Closed within tolerance", str(check.passed_count)],
        ["Failed", str(len(check.failures))],
        ["Not applicable", str(len(check.skipped))],
        ["Largest identity residual",
         _verification_number(check.worst_identity_residual, "")],
    ]

    out = [_section(
        "Verification",
        "The report's own arithmetic, re-derived independently and reported "
        "with its residuals.",
    )]
    out.append(_two_up(
        ex.wrap("Arithmetic verification", _table(
            columns="(2.4fr, auto, auto, auto, auto)",
            align="(left, left, right, right, center)",
            header=["Check", "Type", "Residual", "Tolerance", "Result"],
            rows=rows,
            size=tokens.SIZE_SMALL,
        ), "Computed by GrowNXT. An identity must close to floating-point "
           "precision because it is a definition rather than an estimate. A "
           "reconciliation compares a figure computed here against the "
           "provider's published value for the same quantity and is allowed "
           "the rounding the provider's own precision implies. A guardrail "
           "asserts an editorial rule held in practice."),
        ex.wrap("Verification summary", _panel("RESULT", summary),
                "A check with nothing to compare is recorded as not "
                "applicable rather than as a pass."),
        ratio="(2.5fr, 1fr)",
    ))

    if check.failures:
        for failure in check.failures:
            out.append(_note_block(
                "%s. Expected %s, computed %s, a residual of %s against a "
                "tolerance of %s. %s" % (
                    failure.name,
                    _verification_number(failure.expected, failure.unit),
                    _verification_number(failure.actual, failure.unit),
                    _verification_number(failure.delta, failure.unit),
                    _verification_number(failure.tolerance, failure.unit),
                    failure.detail,
                ), tone="alert"))
    else:
        out.append(_note_block(
            "Every applicable check closed within tolerance. The identities "
            "that the capital-allocation, DuPont and composite exhibits rest "
            "on hold to floating-point precision: the cash flow statement "
            "articulates to its own reported movement in cash, sources equal "
            "uses, the five DuPont factors multiply to return on equity, and "
            "each composite total equals the sum of the components printed "
            "beside it."))

    # The composite notes are deliberately NOT repeated here. They already
    # appear twice: once in the cover's limitations band as a document-level
    # summary, and once as a note on the exhibit whose framework was
    # withheld, which is where a reader meets the gap in context. A third
    # copy in the verification table would be noise.
    return "".join(out)


def _verification_number(value: Optional[float], unit: str) -> str:
    """Formats a residual or tolerance with the unit it is measured in.

    Residuals span many orders of magnitude - a floating-point identity
    closes near ten to the minus fifteen while a monetary tolerance is half a
    crore - so a fixed number of decimals would render almost all of them as
    0.0000, which is a claim of exactness the figures do not make. Small
    magnitudes therefore switch to scientific notation, which is unusual in a
    financial table and exactly right in a verification appendix: the reader
    needs to see that a residual is at machine precision rather than merely
    small, and that a tolerance of ten to the minus six was genuinely the
    specification rather than a rounded nil.
    """
    if value is None:
        return fmt.DASH
    suffix = (" " + unit) if unit else ""
    if value == 0:
        return "0" + suffix
    if abs(value) < 1e-4:
        # Parenthesised for negatives like every other figure in the report.
        # A bare minus sign here would have left one column carrying two
        # different conventions for the same thing.
        magnitude = "{:.1e}".format(abs(value))
        return ("(" + magnitude + ")" if value < 0 else magnitude) + suffix
    return fmt.num(value, 4) + suffix


def disclaimer(as_of: str) -> str:
    """Renders the closing disclosure block."""
    lines = [
        "This report is generated automatically by %s from third-party data and is "
        "provided for information purposes only." % tokens.BRAND_NAME,
        "It is not investment advice, nor an offer or solicitation to buy or sell any "
        "security. No rating, target price or forward estimate is expressed or implied, "
        "and nothing in this document should be read as a recommendation.",
        "Figures are as reported by the underlying data provider as of %s and may be "
        "restated, revised or incomplete. Margins, growth rates, returns on capital, "
        "leverage, working-capital, earnings-quality, per-share, enterprise-value, "
        "capital-allocation and reinvestment measures are computed by %s from that "
        "data." % (as_of, tokens.BRAND_NAME),
        "The Piotroski F-Score and Altman Z-Score are published third-party "
        "frameworks applied here to that data; their coefficients and thresholds are "
        "the original authors' and are not fitted or optimised. Each is presented "
        "with every component that produced it, and each is withheld where the "
        "framework does not apply to the company in question. A composite score is "
        "a summary of the sub-tests beside it and is not a rating, a recommendation "
        "or a prediction of default.",
        "The share price shown is inferred from the trailing price-to-earnings ratio and "
        "trailing earnings per share. It is a derivation, not a market quotation, and "
        "will differ from the traded price.",
        "Past performance is not indicative of future results. Recipients should conduct "
        "their own analysis and consult a licensed financial adviser before making any "
        "investment decision. %s accepts no liability for any loss arising from use of "
        "this report." % tokens.BRAND_NAME,
    ]
    body = " ".join(fmt.escape_typst(line) for line in lines)
    return "\n".join([
        "#v(8pt)",
        "#line(length: 100%%, stroke: 0.8pt + %s)" % C_BRAND,
        "#v(3pt)",
        "#text(size: %s, weight: \"semibold\", fill: %s, tracking: 0.4pt)"
        "[IMPORTANT DISCLOSURES AND DISCLAIMER]" % (tokens.SIZE_SMALL, C_BRAND),
        "#v(2.5pt)",
        "#block[#set par(justify: true, leading: 0.48em)",
        " #text(size: %s, fill: %s)[%s]]" % (tokens.SIZE_FOOTNOTE, C_MUTED, body),
        "#v(3pt)",
        "#text(size: %s, fill: %s)[Source data: Financial Data Collector REST service. "
        "Report generated %s.]" % (tokens.SIZE_FOOTNOTE, C_MUTED, fmt.escape_typst(as_of)),
    ]) + "\n"


def preamble(snap: CompanySnapshot, as_of: str) -> str:
    """Page setup, running header and footer, and base text style."""
    return "\n".join([
        "#set page(",
        "  paper: \"%s\"," % tokens.PAGE_SIZE,
        "  margin: (top: %s, bottom: %s, x: %s)," % (
            tokens.MARGIN_TOP, tokens.MARGIN_BOTTOM, tokens.MARGIN_X),
        "  header: context {",
        "    set text(size: %s, fill: %s)" % (tokens.SIZE_SMALL, C_MUTED),
        "    grid(columns: (1fr, auto), align: (left + bottom, right + bottom),",
        "      [#text(font: %s, size: 12.5pt, weight: \"bold\", fill: %s)[%s]"
        " #h(3.5pt) #text(size: %s, fill: %s, tracking: 0.5pt)[%s]]," % (
            _font_list(tokens.FONT_DISPLAY), C_BRAND, tokens.BRAND_NAME,
            tokens.SIZE_FOOTNOTE, C_MUTED, tokens.BRAND_TAGLINE.upper()),
        "      [#text(weight: \"semibold\", fill: %s)[%s] #sym.dot.c %s])" % (
            C_INK, fmt.escape_typst(snap.ticker), fmt.escape_typst(as_of)),
        "    v(-2.5pt)",
        "    line(length: 100%%, stroke: 0.9pt + %s)" % C_BRAND,
        "  },",
        "  footer: context {",
        "    set text(size: %s, fill: %s)" % (tokens.SIZE_FOOTNOTE, C_MUTED),
        "    line(length: 100%%, stroke: 0.4pt + %s)" % C_RULE,
        "    v(1.5pt)",
        "    let current = counter(page).get().first()",
        "    let total = counter(page).final().first()",
        "    grid(columns: (1fr, auto), align: (left, right),",
        "      [%s #sym.dot.c For information only, not investment advice]," % (
            fmt.escape_typst(tokens.BRAND_NAME)),
        "      [Page #current of #total])",
        "  },",
        ")",
        "#set text(font: %s, size: %s, fill: %s, number-width: \"tabular\", "
        "number-type: \"lining\")" % (
            _font_list(tokens.FONT_BODY), tokens.SIZE_BODY, C_INK),
        "#set par(justify: false, leading: 0.52em)",
        "#show table.cell.where(y: 0): set text(size: %s)" % tokens.SIZE_SMALL,
    ]) + "\n"


def build_document(
    snap: CompanySnapshot,
    derived: DerivedAnalytics,
    comp: Composites,
    check: SelfCheck,
    charts: Dict[str, Path],
    as_of: Optional[str] = None,
) -> str:
    """Assembles the complete Typst source for one report.

    Section order follows the argument the report is making rather than the
    order the data arrives in. Earnings power and near-term trajectory
    establish what the business earns; returns on capital and financial
    position establish what it earns that on and how it is funded; cash
    generation tests whether the profit is real; capital allocation traces
    where that cash went and whether the reinvested share earned its keep;
    the composite scores then pass two standard frameworks over everything
    established so far, which is why they cannot come earlier. Valuation
    prices the result, ownership says who holds it, and verification shows
    the arithmetic.

    Args:
        snap: Populated company snapshot.
        derived: Tier 1 computed analytics.
        comp: Tier 2 composites.
        check: Arithmetic self-verification of the two above.
        charts: Mapping of chart key to SVG path. Missing keys are laid out
            around rather than left as gaps.
        as_of: Display date. Defaults to today.

    Returns:
        Typst source ready to compile.
    """
    stamp = as_of or date.today().strftime("%d %b %Y")
    ex = Exhibits()
    sections = [
        earnings_power,
        near_term,
        returns_on_capital,
        financial_position,
        cash_and_quality,
        capital_allocation,
        composite_scores,
        valuation,
        shareholder_returns,
        verification,
    ]
    parts = [preamble(snap, stamp), cover(snap, derived, comp, stamp)]
    parts += [builder(snap, derived, comp, check, ex, charts)
              for builder in sections]
    parts.append(disclaimer(stamp))
    return "\n".join(parts)
