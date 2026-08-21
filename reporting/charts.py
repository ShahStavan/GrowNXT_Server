"""Chart rendering for GrowNXT reports.

Every chart is emitted as SVG so that Typst embeds it as vector artwork and
it stays sharp in print. Text is converted to paths on save, which makes
output independent of the fonts installed on the rendering host.

House style, applied through `tokens.matplotlib_rc`:

    - Charts carry NO title. The Typst exhibit caption titles them, so the
      document's own typeface and hierarchy apply rather than matplotlib's.
      A chart title set in a different face is the clearest sign that a
      figure was pasted into a report rather than designed with it.
    - Bars always start at zero. A truncated bar axis misstates magnitude.
    - Horizontal gridlines only, hairline, behind the data. No left spine:
      the gridlines already carry the scale.
    - The latest period is emphasised, since that is what a reader looks
      for first.
    - Series colours come from a stepped-lightness ramp, so ordering
      survives greyscale printing.
    - No label may overlap another. Where a scatter would collide, the
      chart form is changed rather than the labels shrunk.
"""

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from reporting import composites as composites_module
from reporting import fmt, tokens
from reporting.analytics import DerivedAnalytics
from reporting.composites import Composites
from reporting.snapshot import CompanySnapshot

logger = logging.getLogger(__name__)

plt.rcParams.update(tokens.matplotlib_rc())


def _compact(value: float, _pos: int = 0) -> str:
    """Axis tick formatter that abbreviates thousands as 'k'."""
    if abs(value) >= 1000:
        return "{:,.0f}k".format(value / 1000.0)
    return "{:,.0f}".format(value)


def _canvas(
    labels: Sequence[str],
    width: float = tokens.CHART_W_HALF,
    height: float = tokens.CHART_H_STD,
):
    """Opens a figure and returns it with its axis and the period positions.

    Every chart here plots against a period index rather than a date, so the
    positions are always `0..n-1` for the labels being shown.

    Returns:
        The figure, its axis, and the x positions the labels sit at.
    """
    fig, axis = plt.subplots(figsize=(width, height))
    return fig, axis, list(range(len(labels)))


def _period_ticks(
    axis,
    positions: Sequence[int],
    labels: Sequence[str],
    size: Optional[float] = tokens.TICK_PERIOD,
    **kwargs,
) -> None:
    """Places the period labels on the x axis.

    Called after the series are drawn, not at figure creation: matplotlib
    autoscales from what has been plotted, and fixing the ticks first would
    make the order a chart is built in matter. Keeping the call last preserves
    that order while still defining "how a period axis is labelled" once.

    Args:
        axis: Axis to label.
        positions: Tick positions, from `_canvas`.
        labels: Period labels, one per position.
        size: Font size; None leaves the rcParams default, which the one
            full-width annual chart relies on.
        **kwargs: Passed through to `set_xticklabels`, for the one chart that
            pins rotation explicitly.
    """
    axis.set_xticks(list(positions))
    if size is None:
        axis.set_xticklabels(labels, **kwargs)
    else:
        axis.set_xticklabels(labels, fontsize=size, **kwargs)


def _save(fig: "plt.Figure", path: Path) -> Path:
    """Writes a figure to SVG and closes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return path


def _bar_values(values: Sequence[Optional[float]]) -> List[float]:
    """Converts a series to bar heights, mapping absent values to NaN.

    matplotlib raises on a None height but skips NaN silently, which is the
    behaviour wanted here: a missing period should leave a gap, not a zero
    bar that would misstate the figure as nil.
    """
    return [float("nan") if v is None else float(v) for v in values]


def _present(values: Sequence[Optional[float]]) -> bool:
    """Reports whether a series carries at least one usable value."""
    return any(v is not None for v in values)


def _emphasise_last(bars: Sequence, colour: str) -> None:
    """Darkens the final bar so the latest period reads first."""
    for bar in list(bars)[-1:]:
        bar.set_color(colour)


def _seat_overlay(
    axis: "plt.Axes",
    values: Sequence[float],
    floor_zero: bool = False,
) -> None:
    """Sets a twin axis so its line sits above the bars beneath it.

    Multiplicative padding flattens a line whose range is narrow. Padding
    by the series' own span keeps the shape readable while reserving the
    lower part of the frame for the bars it overlays.
    """
    usable = [v for v in values if v is not None]
    if not usable:
        return
    low, high = min(usable), max(usable)
    span = (high - low) or (abs(high) * 0.12) or 1.0
    bottom = 0.0 if floor_zero else low - span * 1.25
    axis.set_ylim(bottom, high + span * 0.50)


def _label_ends(
    axis: "plt.Axes",
    positions: Sequence[float],
    values: Sequence[Optional[float]],
    decimals: int = 0,
) -> None:
    """Labels only the first and last points of a series.

    Labelling every point on a dense series turns the chart into a table.
    The endpoints carry the reader from "where it started" to "where it is",
    which is what the chart is for.
    """
    usable = [(p, v) for p, v in zip(positions, values) if v is not None and v == v]
    if not usable:
        return
    for index in {0, len(usable) - 1}:
        position, value = usable[index]
        axis.annotate(
            "{:,.{d}f}".format(value, d=decimals),
            xy=(position, value), xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", fontsize=tokens.DATA_LABEL, color=tokens.INK,
            fontweight="semibold" if index else "normal", zorder=6,
            bbox=dict(boxstyle="square,pad=0.14", facecolor=tokens.SURFACE,
                      edgecolor="none", alpha=0.86),
        )


def annual_revenue_profit(snap: CompanySnapshot, out_dir: Path) -> Optional[Path]:
    """Annual revenue and EBIT bars with the EBIT margin overlaid.

    Combining the absolute and the ratio in one frame is how broker notes
    show whether growth came with or without margin.
    """
    years = [y for y in snap.years if y.revenue is not None][-tokens.DISPLAY_YEARS:]
    if len(years) < 2:
        return None

    labels = [fmt.period_label(y.period) for y in years]
    revenue = [y.revenue for y in years]
    ebit = [y.ebit for y in years]
    margins = [y.ebit_margin for y in years]

    fig, axis, positions = _canvas(labels, width=tokens.CHART_W_FULL)
    width = 0.36

    bars_rev = axis.bar([p - width / 2 for p in positions], _bar_values(revenue),
                        width, color=tokens.SERIES[2], label="Revenue", zorder=3)
    bars_ebit = axis.bar([p + width / 2 for p in positions], _bar_values(ebit),
                         width, color=tokens.SERIES[4], label="EBIT", zorder=3)
    _emphasise_last(bars_rev, tokens.SERIES[0])
    _emphasise_last(bars_ebit, tokens.SERIES[3])

    _period_ticks(axis, positions, labels, size=None)
    axis.set_ylabel("Rs cr")
    axis.yaxis.set_major_formatter(FuncFormatter(_compact))
    axis.set_axisbelow(True)
    top = max([v for v in revenue if v is not None] or [0])
    if top:
        axis.set_ylim(top=top * 1.20)

    twin = axis.twinx()
    twin.plot(positions, _bar_values(margins), color=tokens.ACCENT_LINE,
              linewidth=1.3, marker="o", markersize=2.8,
              markerfacecolor=tokens.SURFACE, markeredgewidth=1.0,
              zorder=5, label="EBIT margin (RHS)")
    twin.set_ylabel("EBIT margin %", color=tokens.ACCENT_LINE)
    twin.tick_params(axis="y", colors=tokens.ACCENT_LINE)
    twin.grid(False)
    twin.spines["top"].set_visible(False)
    twin.spines["left"].set_visible(False)
    _label_ends(twin, positions, margins, decimals=1)
    _seat_overlay(twin, [m for m in margins if m is not None])

    handles, names = axis.get_legend_handles_labels()
    h2, n2 = twin.get_legend_handles_labels()
    axis.legend(handles + h2, names + n2, loc="upper left",
                bbox_to_anchor=(0.0, 1.10), ncol=3)

    return _save(fig, out_dir / "annual_revenue_profit.svg")


def quarterly_trend(snap: CompanySnapshot, out_dir: Path) -> Optional[Path]:
    """Eight-quarter revenue bars with the PAT margin overlaid."""
    quarters = [q for q in snap.quarters if q.revenue is not None][-tokens.DISPLAY_QUARTERS:]
    if len(quarters) < 2:
        return None

    labels = [fmt.period_label(q.period) for q in quarters]
    revenue = [q.revenue for q in quarters]
    margins = [q.pat_margin for q in quarters]

    fig, axis, positions = _canvas(labels)
    bars = axis.bar(positions, _bar_values(revenue), 0.58,
                    color=tokens.SERIES[3], zorder=3, label="Revenue")
    _emphasise_last(bars, tokens.SERIES[1])

    _period_ticks(axis, positions, labels, size=tokens.TICK_DENSE, rotation=0)
    axis.set_ylabel("Revenue (Rs cr)")
    axis.yaxis.set_major_formatter(FuncFormatter(_compact))
    axis.set_axisbelow(True)
    top = max([v for v in revenue if v is not None] or [0])
    if top:
        axis.set_ylim(top=top * 1.22)

    twin = axis.twinx()
    twin.plot(positions, _bar_values(margins), color=tokens.ACCENT_LINE,
              linewidth=1.3, marker="o", markersize=2.6,
              markerfacecolor=tokens.SURFACE, markeredgewidth=0.9,
              zorder=5, label="PAT margin (RHS)")
    twin.set_ylabel("PAT margin %", color=tokens.ACCENT_LINE)
    twin.tick_params(axis="y", colors=tokens.ACCENT_LINE)
    twin.grid(False)
    twin.spines["top"].set_visible(False)
    twin.spines["left"].set_visible(False)
    _label_ends(twin, positions, margins, decimals=1)
    _seat_overlay(twin, [m for m in margins if m is not None])

    handles, names = axis.get_legend_handles_labels()
    h2, n2 = twin.get_legend_handles_labels()
    axis.legend(handles + h2, names + n2, loc="upper left",
                bbox_to_anchor=(0.0, 1.13), ncol=2)

    return _save(fig, out_dir / "quarterly_trend.svg")


def capital_structure(snap: CompanySnapshot, out_dir: Path) -> Optional[Path]:
    """Equity, debt and cash bars with debt-to-equity overlaid.

    Series absent from the filing are dropped rather than drawn empty.
    Banks, for instance, report no consolidated 'total debt' line — deposits
    and borrowings are separate — so the debt bars and the leverage overlay
    are simply omitted instead of implying a zero.
    """
    rows = [b for b in snap.balance if b.equity is not None][-tokens.DISPLAY_YEARS:]
    if len(rows) < 2:
        return None

    candidates = [
        ("Equity", [b.equity for b in rows], tokens.SERIES[1]),
        ("Debt", [b.debt for b in rows], tokens.SERIES[3]),
        ("Cash", [b.cash for b in rows], tokens.SERIES[4]),
    ]
    series = [(n, v, c) for n, v, c in candidates if _present(v)]
    if not series:
        return None

    labels = [fmt.period_label(b.period) for b in rows]
    fig, axis, positions = _canvas(labels)
    width = 0.74 / len(series)
    offset = -(len(series) - 1) / 2.0

    for index, (name, values, colour) in enumerate(series):
        axis.bar([p + (offset + index) * width for p in positions],
                 _bar_values(values), width, color=colour, label=name, zorder=3)

    _period_ticks(axis, positions, labels)
    axis.set_ylabel("Rs cr")
    axis.yaxis.set_major_formatter(FuncFormatter(_compact))
    axis.set_axisbelow(True)
    top = max([v for _, values, _ in series for v in values if v is not None] or [0])
    if top:
        axis.set_ylim(top=top * 1.24)

    ratios = [b.debt_to_equity for b in rows]
    handles, names = axis.get_legend_handles_labels()
    if _present(ratios):
        twin = axis.twinx()
        twin.plot(positions, _bar_values(ratios), color=tokens.ACCENT_LINE,
                  linewidth=1.3, marker="o", markersize=2.6,
                  markerfacecolor=tokens.SURFACE, markeredgewidth=0.9,
                  zorder=5, label="D/E (RHS)")
        twin.set_ylabel("Debt / equity (x)", color=tokens.ACCENT_LINE)
        twin.tick_params(axis="y", colors=tokens.ACCENT_LINE)
        twin.grid(False)
        twin.spines["top"].set_visible(False)
        twin.spines["left"].set_visible(False)
        _seat_overlay(twin, [r for r in ratios if r is not None], floor_zero=True)
        h2, n2 = twin.get_legend_handles_labels()
        handles, names = handles + h2, names + n2

    axis.legend(handles, names, loc="upper left",
                bbox_to_anchor=(0.0, 1.13), ncol=len(names))

    return _save(fig, out_dir / "capital_structure.svg")


def cash_generation(snap: CompanySnapshot, out_dir: Path) -> Optional[Path]:
    """Operating cash flow, capital expenditure and free cash flow."""
    rows = [c for c in snap.cashflow
            if c.cfo is not None or c.fcf is not None][-tokens.DISPLAY_YEARS:]
    if len(rows) < 2:
        return None

    candidates = [
        ("Operating CF", [c.cfo for c in rows], tokens.SERIES[1]),
        ("Capex", [c.capex for c in rows], tokens.SERIES[3]),
        ("Free CF", [c.fcf for c in rows], tokens.SERIES[4]),
    ]
    series = [(n, v, c) for n, v, c in candidates if _present(v)]
    if not series:
        return None

    labels = [fmt.period_label(c.period) for c in rows]
    fig, axis, positions = _canvas(labels)
    width = 0.74 / len(series)
    offset = -(len(series) - 1) / 2.0

    for index, (name, values, colour) in enumerate(series):
        axis.bar([p + (offset + index) * width for p in positions],
                 _bar_values(values), width, color=colour, label=name, zorder=3)

    _period_ticks(axis, positions, labels)
    axis.set_ylabel("Rs cr")
    axis.yaxis.set_major_formatter(FuncFormatter(_compact))
    axis.set_axisbelow(True)
    axis.axhline(0, color=tokens.RULE, linewidth=0.6, zorder=2)
    values_all = [v for _, vals, _ in series for v in vals if v is not None]
    if values_all:
        span = max(values_all) - min(min(values_all), 0)
        axis.set_ylim(min(min(values_all), 0) - span * 0.06,
                      max(values_all) + span * 0.26)
    axis.legend(loc="upper left", bbox_to_anchor=(0.0, 1.13), ncol=len(series))

    return _save(fig, out_dir / "cash_generation.svg")


def shareholding(snap: CompanySnapshot, out_dir: Path) -> Optional[Path]:
    """Institutional holdings over time, as lines.

    A 0-100 stacked area was tried first and rejected: with a promoter
    block near 72% and barely moving, four fifths of the frame carried no
    information and the institutional detail was compressed into a sliver.
    Foreign, domestic and mutual-fund holdings are what actually move, and
    what accumulation or distribution is read from — so they get the axis
    to themselves. The full mix, promoter included, is in the adjacent
    table where a static figure belongs.
    """
    rows = [h for h in snap.holdings if h.promoter is not None]
    if len(rows) < 2:
        return None

    labels = [h.date[2:7] for h in rows]
    series = [
        ("FII", [h.fii for h in rows], tokens.SERIES[0]),
        ("DII", [h.dii for h in rows], tokens.SERIES[2]),
        ("Mutual funds", [h.mutual_fund for h in rows], tokens.SERIES[3]),
    ]
    series = [(n, v, c) for n, v, c in series if _present(v)]
    if not series:
        return None

    fig, axis, positions = _canvas(labels)

    for name, values, colour in series:
        axis.plot(positions, _bar_values(values), color=colour, linewidth=1.5,
                  marker="o", markersize=2.8, markerfacecolor=tokens.SURFACE,
                  markeredgewidth=0.9, label=name, zorder=4)
        _label_ends(axis, positions, values, decimals=1)

    _period_ticks(axis, positions, labels)
    axis.set_ylabel("Holding %")
    axis.set_xlim(-0.25, len(labels) - 0.75)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", bbox_to_anchor=(0.0, 1.15), ncol=3)
    valid = [v for _, values, _ in series for v in values if v is not None]
    if valid:
        span = (max(valid) - min(valid)) or 1.0
        axis.set_ylim(max(0.0, min(valid) - span * 0.30), max(valid) + span * 0.42)

    return _save(fig, out_dir / "shareholding.svg")


def _ranked_panel(
    axis: "plt.Axes",
    entries: List[Tuple[str, float, bool]],
    reference: Optional[float],
    reference_label: str,
    xlabel: str,
) -> None:
    """Draws one ranked lollipop panel of peer multiples.

    Args:
        axis: Target axes.
        entries: (ticker, value, is_subject) triples, unsorted.
        reference: Industry benchmark to mark, if available.
        reference_label: Caption for the benchmark line.
        xlabel: Axis label.

    A ranked horizontal layout is used in preference to a two-dimensional
    scatter because peers routinely share almost identical multiples — the
    scatter it replaces put HDFC Bank's label on top of Axis Bank's. Here
    each peer owns a row, so labels cannot collide by construction.
    """
    ordered = sorted(entries, key=lambda item: item[1])
    positions = list(range(len(ordered)))
    highest = max(v for _, v, _ in ordered)

    # Reserve a right-hand column for the values before drawing, so labels
    # sit in one aligned rail. Annotating beside each dot instead put them
    # on top of the industry reference line whenever a peer traded near it.
    axis.set_xlim(0, highest * 1.17)
    label_x = highest * 1.16

    for position, (ticker, value, is_subject) in zip(positions, ordered):
        colour = tokens.BRAND if is_subject else tokens.SERIES[2]
        axis.hlines(position, 0, value, color=colour,
                    linewidth=2.6 if is_subject else 1.6,
                    alpha=1.0 if is_subject else 0.55, zorder=3)
        axis.plot([value], [position], marker="o",
                  markersize=5.0 if is_subject else 3.8,
                  color=colour, zorder=4)
        axis.annotate(
            "{:,.1f}".format(value),
            xy=(label_x, position), va="center", ha="right", fontsize=tokens.TICK_RANKED,
            color=tokens.INK if is_subject else tokens.MUTED,
            fontweight="semibold" if is_subject else "normal",
            annotation_clip=False,
        )

    axis.set_yticks(positions)
    axis.set_yticklabels([t for t, _, _ in ordered], fontsize=tokens.TICK_RANKED)
    for label, (_, _, is_subject) in zip(axis.get_yticklabels(), ordered):
        if is_subject:
            label.set_color(tokens.BRAND)
            label.set_fontweight("semibold")

    axis.set_xlabel(xlabel)
    axis.grid(axis="x", color=tokens.RULE_LIGHT, linewidth=0.5)
    axis.grid(axis="y", visible=False)
    axis.set_axisbelow(True)
    axis.spines["bottom"].set_visible(False)
    axis.margins(y=0.14)

    if reference is not None:
        axis.axvline(reference, color=tokens.ACCENT_LINE, linewidth=0.9,
                     linestyle=(0, (3, 2)), zorder=2)
        axis.annotate(
            reference_label + " {:,.1f}".format(reference),
            xy=(reference, axis.get_ylim()[1]), xytext=(3, -6),
            textcoords="offset points", fontsize=tokens.DATA_LABEL_TIGHT,
            color=tokens.ACCENT_LINE, va="top", ha="left",
        )


def peer_valuation(snap: CompanySnapshot, out_dir: Path) -> Optional[Path]:
    """Two ranked panels comparing peer P/E and P/B against the industry.

    Replaces an earlier bubble scatter, which collided labels whenever two
    peers traded on similar multiples.
    """
    pe_entries = [(p.ticker, p.pe, p.is_subject)
                  for p in snap.peers if p.pe is not None and p.ticker]
    pb_entries = [(p.ticker, p.pb, p.is_subject)
                  for p in snap.peers if p.pb is not None and p.ticker]
    if len(pe_entries) < 2 and len(pb_entries) < 2:
        return None

    panels = [entry for entry in (
        (pe_entries, snap.key_ratios.get("indpe"), "Industry", "Trailing P/E (x)"),
        (pb_entries, snap.key_ratios.get("indpb"), "Industry", "Price / book (x)"),
    ) if len(entry[0]) >= 2]

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(tokens.CHART_W_FULL, tokens.CHART_H_STD),
    )
    axis_list = list(axes) if len(panels) > 1 else [axes]

    for axis, (entries, reference, ref_label, xlabel) in zip(axis_list, panels):
        _ranked_panel(axis, entries, reference, ref_label, xlabel)

    return _save(fig, out_dir / "peer_valuation.svg")


def margin_bridge(snap: CompanySnapshot, out_dir: Path) -> Optional[Path]:
    """Quarterly EBIT and PAT margins as paired lines.

    Gives the report a pure-ratio chart. The absolute charts answer "how
    big"; this one answers "how profitable", which is the question a
    margin-driven business is actually judged on.
    """
    quarters = [q for q in snap.quarters if q.revenue][-tokens.DISPLAY_QUARTERS:]
    if len(quarters) < 3:
        return None

    labels = [fmt.period_label(q.period) for q in quarters]
    ebit_margin = [q.ebit_margin for q in quarters]
    pat_margin = [q.pat_margin for q in quarters]
    if not _present(ebit_margin) or not _present(pat_margin):
        return None

    fig, axis, positions = _canvas(labels)

    for values, name, colour in (
        (ebit_margin, "EBIT margin", tokens.SERIES[1]),
        (pat_margin, "PAT margin", tokens.SERIES[3]),
    ):
        axis.plot(positions, _bar_values(values), color=colour, linewidth=1.5,
                  marker="o", markersize=2.8, markerfacecolor=tokens.SURFACE,
                  markeredgewidth=0.9, label=name, zorder=4)
        _label_ends(axis, positions, values, decimals=1)

    _period_ticks(axis, positions, labels, size=tokens.TICK_DENSE)
    axis.set_ylabel("Margin %")
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", bbox_to_anchor=(0.0, 1.13), ncol=2)
    valid = [v for v in ebit_margin + pat_margin if v is not None]
    if valid:
        span = max(valid) - min(valid) or 1.0
        axis.set_ylim(min(valid) - span * 0.35, max(valid) + span * 0.55)

    return _save(fig, out_dir / "margin_bridge.svg")


def rolling_trend(derived: DerivedAnalytics, out_dir: Path) -> Optional[Path]:
    """Trailing-four-quarter revenue with the trailing EBIT margin.

    A single TTM figure is one dot; rolling the window strips seasonality
    out and shows whether the trailing trend is still improving.
    """
    points = [p for p in derived.rolling if p.revenue is not None][-tokens.DISPLAY_QUARTERS:]
    if len(points) < 3:
        return None

    labels = [p.label for p in points]
    revenue = [p.revenue for p in points]
    margins = [p.ebit_margin for p in points]

    fig, axis, positions = _canvas(labels)
    bars = axis.bar(positions, _bar_values(revenue), 0.58,
                    color=tokens.SERIES[3], zorder=3, label="Trailing revenue")
    _emphasise_last(bars, tokens.SERIES[1])

    _period_ticks(axis, positions, labels, size=tokens.TICK_DENSE)
    axis.set_ylabel("Trailing revenue (Rs cr)")
    axis.yaxis.set_major_formatter(FuncFormatter(_compact))
    axis.set_axisbelow(True)
    top = max([v for v in revenue if v is not None] or [0])
    if top:
        axis.set_ylim(top=top * 1.22)

    twin = axis.twinx()
    twin.plot(positions, _bar_values(margins), color=tokens.ACCENT_LINE,
              linewidth=1.3, marker="o", markersize=2.6,
              markerfacecolor=tokens.SURFACE, markeredgewidth=0.9,
              zorder=5, label="Trailing EBIT margin (RHS)")
    twin.set_ylabel("EBIT margin %", color=tokens.ACCENT_LINE)
    twin.tick_params(axis="y", colors=tokens.ACCENT_LINE)
    twin.grid(False)
    twin.spines["top"].set_visible(False)
    twin.spines["left"].set_visible(False)
    _label_ends(twin, positions, margins, decimals=1)
    _seat_overlay(twin, [m for m in margins if m is not None])

    handles, names = axis.get_legend_handles_labels()
    h2, n2 = twin.get_legend_handles_labels()
    axis.legend(handles + h2, names + n2, loc="upper left",
                bbox_to_anchor=(0.0, 1.13), ncol=2)

    return _save(fig, out_dir / "rolling_trend.svg")


def returns_trend(derived: DerivedAnalytics, out_dir: Path) -> Optional[Path]:
    """Return on equity, capital employed and invested capital over time.

    Plotted together because they answer one question from three angles:
    how much the business earns on the money tied up in it. Divergence is
    itself the signal — ROE rising while ROIC falls means leverage, not
    operating improvement.
    """
    rows = [r for r in derived.annual
            if r.period.strip().upper() != "TTM"][-tokens.DISPLAY_YEARS:]
    if len(rows) < 3:
        return None

    labels = [fmt.period_label(r.period) for r in rows]
    candidates = [
        ("ROE", [r.roe for r in rows], tokens.SERIES[0]),
        ("ROCE", [r.roce for r in rows], tokens.SERIES[2]),
        ("ROIC", [r.roic for r in rows], tokens.SERIES[3]),
    ]
    series = [(n, v, col) for n, v, col in candidates if _present(v)]
    if not series:
        return None

    fig, axis, positions = _canvas(labels)
    for name, values, colour in series:
        axis.plot(positions, _bar_values(values), color=colour, linewidth=1.5,
                  marker="o", markersize=2.8, markerfacecolor=tokens.SURFACE,
                  markeredgewidth=0.9, label=name, zorder=4)
        _label_ends(axis, positions, values, decimals=1)

    _period_ticks(axis, positions, labels)
    axis.set_ylabel("Return %")
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", bbox_to_anchor=(0.0, 1.14), ncol=3)
    valid = [v for _, values, _ in series for v in values if v is not None]
    if valid:
        span = (max(valid) - min(valid)) or 1.0
        axis.set_ylim(max(0.0, min(valid) - span * 0.28), max(valid) + span * 0.44)

    return _save(fig, out_dir / "returns_trend.svg")


def working_capital(derived: DerivedAnalytics, out_dir: Path) -> Optional[Path]:
    """Receivable, inventory and payable days with the resulting cycle.

    Rendered only where the provider's cost base supports a day count. For
    a services business the inventory and payable legs are withheld
    upstream, so this chart is skipped rather than drawn half-empty.
    """
    rows = [r for r in derived.annual
            if r.cash_conversion_cycle is not None
            and r.period.strip().upper() != "TTM"][-tokens.DISPLAY_YEARS:]
    if len(rows) < 3:
        return None

    labels = [fmt.period_label(r.period) for r in rows]
    fig, axis, positions = _canvas(labels)
    width = 0.26

    for index, (name, values, colour) in enumerate([
        ("DSO", [r.dso for r in rows], tokens.SERIES[1]),
        ("DIO", [r.dio for r in rows], tokens.SERIES[3]),
        ("DPO", [r.dpo for r in rows], tokens.SERIES[4]),
    ]):
        axis.bar([p + (index - 1) * width for p in positions],
                 _bar_values(values), width, color=colour, label=name, zorder=3)

    _period_ticks(axis, positions, labels)
    axis.set_ylabel("Days")
    axis.set_axisbelow(True)
    top = max([v for r in rows for v in (r.dso, r.dio, r.dpo) if v is not None] or [0])
    if top:
        axis.set_ylim(top=top * 1.32)

    twin = axis.twinx()
    cycle = [r.cash_conversion_cycle for r in rows]
    twin.plot(positions, _bar_values(cycle), color=tokens.ACCENT_LINE,
              linewidth=1.4, marker="o", markersize=2.8,
              markerfacecolor=tokens.SURFACE, markeredgewidth=0.9,
              zorder=5, label="Cash cycle (RHS)")
    twin.axhline(0, color=tokens.RULE, linewidth=0.6)
    twin.set_ylabel("Cash cycle, days", color=tokens.ACCENT_LINE)
    twin.tick_params(axis="y", colors=tokens.ACCENT_LINE)
    twin.grid(False)
    twin.spines["top"].set_visible(False)
    twin.spines["left"].set_visible(False)
    _label_ends(twin, positions, cycle, decimals=0)

    handles, names = axis.get_legend_handles_labels()
    h2, n2 = twin.get_legend_handles_labels()
    axis.legend(handles + h2, names + n2, loc="upper left",
                bbox_to_anchor=(0.0, 1.15), ncol=4)

    return _save(fig, out_dir / "working_capital.svg")


def piotroski_trend(comp: Composites, out_dir: Path) -> Optional[Path]:
    """The F-Score as a series, against the ceiling it is scored out of.

    Bars rather than a line: the score is a count of signals passed, and a
    line implies the intermediate values mean something. The ceiling is drawn
    explicitly because a score is unreadable without its denominator - six
    out of nine and six out of seven are different statements, and for a
    financial two signals are withheld, so the ceiling itself moves.
    """
    points = [p for p in comp.piotroski_trend
              if p.score is not None][-tokens.DISPLAY_YEARS:]
    if len(points) < 3:
        return None

    labels = [fmt.period_label(p.period) for p in points]
    scores = [p.score for p in points]
    ceiling = max(p.computable for p in points)

    fig, axis, positions = _canvas(labels)
    bars = axis.bar(positions, _bar_values(scores), 0.62,
                    color=tokens.SERIES[2], zorder=3)
    _emphasise_last(bars, tokens.SERIES[0])

    # Piotroski reads eight and above as strong and two and below as weak.
    # Shading those bands puts the score in the author's own terms rather
    # than leaving the reader to remember them.
    if ceiling >= 8:
        axis.axhspan(8, ceiling, color=tokens.POSITIVE, alpha=0.07, zorder=1)
    axis.axhspan(0, 2, color=tokens.NEGATIVE, alpha=0.07, zorder=1)
    axis.axhline(ceiling, color=tokens.FAINT, linewidth=0.7,
                 linestyle=(0, (2.5, 2)), zorder=2)
    axis.annotate("%d signals evaluated" % ceiling,
                  xy=(len(positions) - 0.5, ceiling), xytext=(0, 2),
                  textcoords="offset points", ha="right", va="bottom",
                  fontsize=tokens.ANNOTATION, color=tokens.MUTED, zorder=6)

    for position, score in zip(positions, scores):
        axis.annotate("%d" % score, xy=(position, score), xytext=(0, 2.5),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize=tokens.SCORE_VALUE, color=tokens.INK, fontweight="semibold",
                      zorder=6)

    _period_ticks(axis, positions, labels)
    axis.set_ylabel("Signals passed")
    axis.set_ylim(0, ceiling + 1.35)
    axis.set_yticks(list(range(0, ceiling + 1, 2)))
    axis.set_axisbelow(True)

    return _save(fig, out_dir / "piotroski_trend.svg")


def altman_zones(comp: Composites, out_dir: Path) -> Optional[Path]:
    """The Z-prime series against Altman's own distress and safe bands.

    This is the one place in the report where a chart carries information a
    table cannot: the score means nothing without its thresholds, and a
    threshold is a position on an axis. Shading the three zones puts every
    year in its band at a glance, which a column of numbers cannot do.
    """
    points = [p for p in comp.altman_trend
              if p.score is not None][-tokens.DISPLAY_YEARS:]
    if len(points) < 3:
        return None

    labels = [fmt.period_label(p.period) for p in points]
    scores = [p.score for p in points]
    safe = composites_module.ALTMAN_PRIVATE_SAFE
    distress = composites_module.ALTMAN_PRIVATE_DISTRESS

    fig, axis, positions = _canvas(labels)

    low = min(min(scores), distress) - 0.55
    high = max(max(scores), safe) + 0.45
    axis.axhspan(low, distress, color=tokens.NEGATIVE, alpha=0.08, zorder=1)
    axis.axhspan(distress, safe, color=tokens.FAINT, alpha=0.10, zorder=1)
    axis.axhspan(safe, high, color=tokens.POSITIVE, alpha=0.08, zorder=1)

    for boundary in (distress, safe):
        axis.axhline(boundary, color=tokens.FAINT, linewidth=0.6,
                     linestyle=(0, (2.5, 2)), zorder=2)

    # Band names sit in the vertical middle of their own band, against the
    # right edge, because the series enters from the left. An earlier version
    # placed these in data coordinates outside the axes, where matplotlib
    # clipped them away silently. The blended transform takes x as an axes
    # fraction and y as a data value, which puts them inside the frame at the
    # right height regardless of how the y-axis is scaled.
    bands = (
        ((low + distress) / 2.0, "Distress"),
        ((distress + safe) / 2.0, "Grey"),
        ((safe + high) / 2.0, "Safe"),
    )
    # Anchored left, not right: the right edge already carries the closing
    # value label, and at TCS the 'Safe' band name landed on top of it.
    for centre, name in bands:
        axis.text(0.015, centre, name, transform=axis.get_yaxis_transform(),
                  ha="left", va="center", fontsize=tokens.ANNOTATION, color=tokens.MUTED,
                  zorder=6)

    axis.plot(positions, _bar_values(scores), color=tokens.SERIES[0],
              linewidth=1.5, marker="o", markersize=2.8,
              markerfacecolor=tokens.SURFACE, markeredgewidth=0.9, zorder=4)
    _label_ends(axis, positions, scores, decimals=2)

    _period_ticks(axis, positions, labels)
    axis.set_ylabel("Z-prime score")
    axis.set_ylim(low, high)
    axis.set_axisbelow(True)
    axis.grid(False)

    return _save(fig, out_dir / "altman_zones.svg")


def sources_and_uses(comp: Composites, out_dir: Path) -> Optional[Path]:
    """Where the cash came from and where it went, as matched compositions.

    Two bars of identical length, because sources equal uses by construction.
    That equality is the point of the exhibit and a stacked pair states it
    visually in a way a two-column table cannot. Scaled to percentages
    rather than absolutes for the same reason: the question here is the mix,
    and the accompanying table carries the amounts.
    """
    allocation = comp.sources_uses
    total = allocation.total_sources
    if not allocation.sources or not allocation.uses or not total:
        return None

    fig, axis = plt.subplots(figsize=(tokens.CHART_W_FULL, 1.46))

    # A shared ramp across both bars would imply that the third source and
    # the third use are related. They are not, so each bar walks the ramp
    # independently and the segment labels carry the meaning.
    for row, items in enumerate((allocation.uses, allocation.sources)):
        left = 0.0
        for index, item in enumerate(items):
            share = fmt.pos_margin(item.magnitude, total)
            if share is None:
                continue
            colour = tokens.SERIES[index % len(tokens.SERIES)]
            axis.barh(row, share, 0.55, left=left, color=colour, zorder=3)
            # The ramp darkens toward index zero, so light text is only
            # legible on the first two steps.
            ink = tokens.SURFACE if index < 2 else tokens.INK
            caption = _segment_caption(item.label, share)
            if caption:
                axis.annotate(
                    caption, xy=(left + share / 2.0, row),
                    ha="center", va="center", fontsize=tokens.SEGMENT_CAPTION, color=ink,
                    linespacing=1.15, zorder=6)
            left += share

    axis.set_yticks([0, 1])
    axis.set_yticklabels(["Uses", "Sources"], fontsize=tokens.TICK_CATEGORY,
                         fontweight="semibold")
    axis.set_xlim(0, 100)
    axis.set_xlabel("Share of total cash flows over the window (%)")
    axis.set_ylim(-0.5, 1.5)
    axis.grid(False)
    axis.tick_params(axis="x", labelsize=6.0)
    for spine in ("top", "right", "left", "bottom"):
        axis.spines[spine].set_visible(False)

    return _save(fig, out_dir / "sources_and_uses.svg")


# A stacked segment can hold three lines of 5.5pt text and no more. The
# percentage always takes one, which leaves two for the name. Below the first
# threshold a segment is too narrow for even the percentage; between the two
# it carries the percentage alone.
SEGMENT_MIN_PCT: float = 5.0
SEGMENT_NAMED_PCT: float = 18.0


def _segment_caption(label: str, share: float) -> str:
    """Builds the in-segment caption that fits the space available.

    Written after an earlier version overflowed: a four-line caption in a
    three-line bar put "Net disposals and investment maturities" through the
    segment's own top edge. A caption that does not fit is reduced to the
    percentage rather than drawn over its neighbours, and the source line
    tells the reader the segments run in table order so a bare percentage is
    still identifiable.
    """
    if share < SEGMENT_MIN_PCT:
        return ""
    percentage = "%.0f%%" % share
    if share < SEGMENT_NAMED_PCT:
        return percentage
    name = _wrap_label(label, width=20, max_lines=2)
    if name.endswith("..."):
        return percentage
    return name + "\n" + percentage


def _wrap_label(label: str, width: int = 18, max_lines: int = 3) -> str:
    """Breaks a flow label onto a few lines so it fits inside a bar segment.

    Truncation is marked rather than silent. A two-line version of this
    rendered "Net debt repaid and shares bought back" as ending in "bought",
    which is not an abbreviation of the label but a different statement.
    """
    words = label.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1] + "..."
    return "\n".join(lines)


def reinvestment_identity(comp: Composites, out_dir: Path) -> Optional[Path]:
    """Growth the reinvestment implies, against the growth delivered.

    The bars are what the identity predicts from capital actually put in;
    the line is what revenue did. The distance between them is the exhibit's
    whole subject, so both belong in one frame at one scale.
    """
    reinvestment = comp.reinvestment
    if reinvestment.withheld_reason:
        return None
    rows = [y for y in reinvestment.years
            if y.implied_growth is not None or y.revenue_growth is not None]
    if len(rows) < 3:
        return None

    labels = [fmt.period_label(y.period) for y in rows]
    implied = [y.implied_growth for y in rows]
    actual = [y.revenue_growth for y in rows]

    fig, axis, positions = _canvas(labels)
    axis.bar(positions, _bar_values(implied), 0.58, color=tokens.SERIES[3],
             label="Implied by reinvestment", zorder=3)
    axis.plot(positions, _bar_values(actual), color=tokens.ACCENT_LINE,
              linewidth=1.4, marker="o", markersize=2.6,
              markerfacecolor=tokens.SURFACE, markeredgewidth=0.9,
              label="Revenue growth delivered", zorder=5)

    axis.axhline(0, color=tokens.RULE, linewidth=0.6, zorder=2)
    _period_ticks(axis, positions, labels)
    axis.set_ylabel("Growth, % a year")
    axis.set_axisbelow(True)
    values = [v for v in implied + actual if v is not None]
    if values:
        span = (max(values) - min(min(values), 0.0)) or 1.0
        axis.set_ylim(min(min(values), 0.0) - span * 0.10,
                      max(values) + span * 0.30)
    axis.legend(loc="upper left", bbox_to_anchor=(0.0, 1.16), ncol=2)

    return _save(fig, out_dir / "reinvestment_identity.svg")


def dupont_indexed(comp: Composites, out_dir: Path) -> Optional[Path]:
    """The five DuPont factors indexed to their starting level.

    The factors cannot share a raw axis: a burden ratio sits near 0.8, an
    operating margin near 22, a turnover near 0.7 and a multiplier near 1.6,
    so plotting them together at natural scale hides four of the five.
    Rebasing each to 100 at the window's first year answers the question the
    decomposition exists to answer - which factor moved, and by how much -
    and the table beside it carries the levels.
    """
    rows = [r for r in comp.dupont.years
            if r.period.strip().upper() != "TTM"][-tokens.DISPLAY_YEARS:]
    if len(rows) < 3:
        return None

    labels = [fmt.period_label(r.period) for r in rows]
    families = [
        ("Tax burden", [r.tax_burden for r in rows], tokens.SERIES[0]),
        ("Interest burden", [r.interest_burden for r in rows], tokens.SERIES[1]),
        ("Operating margin", [r.operating_margin for r in rows], tokens.SERIES[2]),
        ("Asset turnover", [r.asset_turnover for r in rows], tokens.SERIES[3]),
        ("Equity multiplier", [r.equity_multiplier for r in rows], tokens.SERIES[4]),
    ]

    fig, axis, positions = _canvas(labels)
    plotted = 0
    for name, values, colour in families:
        base = values[0]
        # An index needs a strictly positive base; without one the factor is
        # dropped rather than rebased against something meaningless.
        indexed = [fmt.pos_margin(v, base) for v in values]
        if not _present(indexed):
            continue
        plotted += 1
        axis.plot(positions, _bar_values(indexed), color=colour, linewidth=1.4,
                  marker="o", markersize=2.2, markerfacecolor=tokens.SURFACE,
                  markeredgewidth=0.7, label=name, zorder=4)

    if plotted < 2:
        plt.close(fig)
        return None

    axis.axhline(100.0, color=tokens.RULE, linewidth=0.7, zorder=2)
    _period_ticks(axis, positions, labels)
    axis.set_ylabel("Indexed, %s = 100" % labels[0])
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", bbox_to_anchor=(0.0, 1.30), ncol=3)

    return _save(fig, out_dir / "dupont_indexed.svg")


def render_all(
    snap: CompanySnapshot,
    derived: DerivedAnalytics,
    comp: Composites,
    out_dir: Path,
) -> dict:
    """Renders every chart the data can support.

    Args:
        snap: Populated company snapshot.
        derived: Computed analytics, for the derived-series charts.
        comp: Tier 2 composites, for the scored and identity charts.
        out_dir: Directory for SVG output.

    Returns:
        Mapping of chart key to path, omitting charts with insufficient
        data so the document can lay out around what actually exists.
    """
    snapshot_builders = {
        "annual": annual_revenue_profit,
        "quarterly": quarterly_trend,
        "margins": margin_bridge,
        "capital": capital_structure,
        "cash": cash_generation,
        "holding": shareholding,
        "peers": peer_valuation,
    }
    derived_builders = {
        "rolling": rolling_trend,
        "returns": returns_trend,
        "workcap": working_capital,
    }
    composite_builders = {
        "fscore": piotroski_trend,
        "altman": altman_zones,
        "allocation": sources_and_uses,
        "reinvestment": reinvestment_identity,
        "dupont": dupont_indexed,
    }
    sources = {}
    sources.update({key: snap for key in snapshot_builders})
    sources.update({key: derived for key in derived_builders})
    sources.update({key: comp for key in composite_builders})
    builders = {}
    builders.update(snapshot_builders)
    builders.update(derived_builders)
    builders.update(composite_builders)

    produced = {}
    for key, builder in builders.items():
        try:
            path = builder(sources[key], out_dir)
        except Exception as exc:
            logger.warning("Chart '%s' failed: %s", key, exc, exc_info=True)
            continue
        if path is not None:
            produced[key] = path
        else:
            logger.info("Chart '%s' skipped: insufficient data", key)
    return produced
