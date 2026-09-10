"""Design tokens for GrowNXT institutional reports.

Every colour, size and typeface used by the report originates here. Chart
code and Typst templates both read these values, which is what keeps a
matplotlib figure and the surrounding typeset page looking like one
document rather than two.

Colour policy, in order of importance:

    1. BRAND is never a data colour. It marks the masthead, section rules
       and page furniture only. The moment it encodes a series, the reader
       loses the ability to distinguish branding from meaning.
    2. POSITIVE and NEGATIVE encode direction and nothing else.
    3. The chart series ramp steps lightness rather than hue, so ordering
       survives greyscale printing.

Typeface policy: every face used must have LINING figures. Georgia and
Constantia were both rejected despite being otherwise good print serifs
because they default to old-style (text) figures, where digits sit at
varying heights. In a column of financials that reads as broken. Cambria
and Segoe UI both ship lining figures and support tabular widths.
"""

from typing import Final

# --- Brand ---------------------------------------------------------------

BRAND_NAME: Final[str] = "GrowNXT"
BRAND_TAGLINE: Final[str] = "Equity Research"

# --- Colour --------------------------------------------------------------

# Deep-indigo scheme. Neutrals carry a slight blue bias rather than being
# pure grey, so they read as chosen alongside the brand hue instead of
# inherited.
INK: Final[str] = "#14171F"  # Body text, table figures
INK_SOFT: Final[str] = "#373E4E"  # Secondary text, table headers
MUTED: Final[str] = "#586074"  # Labels, units, source lines
FAINT: Final[str] = "#8A90A2"  # Axis ticks, de-emphasised marks
RULE: Final[str] = "#C0C5D2"  # Table strokes and dividers
RULE_LIGHT: Final[str] = "#E6E9F0"  # Interior hairlines, gridlines
SURFACE: Final[str] = "#FFFFFF"
SUNKEN: Final[str] = "#F3F4F9"  # Zebra and panel fills
SUNKEN_DEEP: Final[str] = "#E8EAF2"  # Table header fill

BRAND: Final[str] = "#1F3A6E"  # Furniture only, never data
BRAND_TINT: Final[str] = "#E9EDF6"

POSITIVE: Final[str] = "#12795C"  # Gains, beats
NEGATIVE: Final[str] = "#B23A3A"  # Declines, misses

# Stepped-lightness ramp built from the brand hue: ordering reads correctly
# in mono as well as colour.
SERIES: Final[list[str]] = [
    "#151F3D",
    "#27407A",
    "#4A66A8",
    "#8598C8",
    "#BFC9E2",
]

# Secondary-axis overlay. Deliberately a warm hue from outside the series
# family, so an overlaid line never reads as one more data series.
ACCENT_LINE: Final[str] = "#A0641C"

# --- Typography ----------------------------------------------------------
# Fallback chains. Typst resolves the first available name; each chain ends
# in a face present on effectively every host. All have lining figures.

FONT_DISPLAY: Final[tuple[str, ...]] = (
    "Cambria",
    "Palatino Linotype",
    "Libertinus Serif",
)
FONT_BODY: Final[tuple[str, ...]] = ("Segoe UI", "Calibri", "DejaVu Sans")
FONT_MONO: Final[tuple[str, ...]] = ("Consolas", "DejaVu Sans Mono")

SIZE_MASTHEAD: Final[str] = "20pt"
SIZE_SUBTITLE: Final[str] = "8.4pt"
SIZE_H1: Final[str] = "11.5pt"  # Section heads
SIZE_H2: Final[str] = "8.0pt"  # Exhibit captions
SIZE_BODY: Final[str] = "8.0pt"
SIZE_TABLE: Final[str] = "7.1pt"
SIZE_PANEL: Final[str] = "7.2pt"
SIZE_SMALL: Final[str] = "6.7pt"
SIZE_FOOTNOTE: Final[str] = "6.0pt"

# --- Page geometry -------------------------------------------------------

PAGE_SIZE: Final[str] = "a4"
MARGIN_TOP: Final[str] = "21mm"
MARGIN_BOTTOM: Final[str] = "15mm"
MARGIN_X: Final[str] = "12mm"

# Gutter between paired half-width exhibits.
GUTTER: Final[str] = "5mm"

# --- display window -------------------------------------------------------
# Charts and tables must cover the SAME periods. The snapshot retains the
# provider's full history so derived metrics have a prior period to work
# against, but showing ten years in a chart beside seven in the adjacent
# table reads as two unrelated documents.

DISPLAY_YEARS: Final[int] = 7
DISPLAY_QUARTERS: Final[int] = 8

# --- Charts --------------------------------------------------------------
# Charts carry no titles: the Typst exhibit caption titles them, so the
# document's own typography applies instead of matplotlib's. Figure widths
# match their display width so no scaling distorts the type.

CHART_DPI: Final[int] = 200
CHART_W_FULL: Final[float] = 7.30
CHART_W_HALF: Final[float] = 3.50
CHART_H_STD: Final[float] = 1.92
CHART_H_SHORT: Final[float] = 1.80


# --- Chart text ----------------------------------------------------------

# Chart text is set in points against the figure rather than in the
# document's type scale, because a chart is sized independently of the text
# block. Each role is named separately even where two share a value today, so
# one can be adjusted without dragging the other with it.

TICK_PERIOD: Final[float] = 5.9  # Period labels on an annual axis
TICK_DENSE: Final[float] = 5.6  # Same, where eight periods crowd the axis
TICK_RANKED: Final[float] = 6.1  # Peer names on a ranked horizontal panel
TICK_CATEGORY: Final[float] = 6.4  # Named categories, e.g. Sources / Uses
DATA_LABEL: Final[float] = 6.0  # Value printed at a series end
DATA_LABEL_TIGHT: Final[float] = 5.7  # Same, where the panel leaves less room
ANNOTATION: Final[float] = 5.6  # Muted explanatory text inside a chart
SEGMENT_CAPTION: Final[float] = 5.5  # Label set inside a composition bar
SCORE_VALUE: Final[float] = 6.2  # Emphasised score printed on a chart


def matplotlib_rc() -> dict:
    """Builds the matplotlib rcParams that mirror these tokens.

    Returns:
        Mapping suitable for `matplotlib.rcParams.update`.
    """
    return {
        "font.family": "sans-serif",
        "font.sans-serif": list(FONT_BODY),
        "font.size": 6.6,
        "figure.dpi": CHART_DPI,
        "savefig.dpi": CHART_DPI,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.transparent": False,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.5,
        "axes.labelsize": 6.4,
        "axes.labelpad": 2.5,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "grid.color": RULE_LIGHT,
        "grid.linewidth": 0.5,
        "xtick.color": FAINT,
        "ytick.color": FAINT,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "xtick.major.size": 0.0,
        "ytick.major.size": 0.0,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.0,
        "legend.frameon": False,
        "legend.fontsize": 6.2,
        "legend.handlelength": 1.1,
        "legend.handleheight": 0.7,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
        "legend.borderpad": 0.0,
        "svg.fonttype": "path",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
    }
