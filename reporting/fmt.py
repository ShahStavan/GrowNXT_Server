"""Number and text formatting rules for GrowNXT reports.

Every figure that reaches a page passes through this module. Centralising it
is what makes precision consistent: ragged decimals across a table are the
clearest signal that a financial document was generated rather than
designed.

Conventions applied here:

    - International digit grouping (1,234,567), never lakh-crore grouping.
      Both reference broker notes use this, and mixing the two within one
      document is worse than either choice.
    - Units are declared in column headers, never repeated per cell.
    - Negatives render in parentheses, the long-standing finance
      convention; a bare minus sign is easy to miss at 7pt.
    - Absent data renders as an em-dash, which must stay visually distinct
      from a real zero. A rendered 0.00 for missing data is a factual error.
"""

from typing import Any

DASH: str = "—"

# Values arriving from the collector are denominated in rupee crore.
UNIT_CRORE: str = "Rs cr"


def _is_number(value: Any) -> bool:
    """Reports whether a value is a usable finite number."""
    if isinstance(value, bool) or value is None:
        return False
    if not isinstance(value, (int, float)):
        return False
    return value == value and value not in (float("inf"), float("-inf"))


def _group(number: float, decimals: int) -> str:
    """Formats a number with international grouping and fixed decimals."""
    return "{:,.{d}f}".format(number, d=decimals)


def _wrap_negative(text: str, negative: bool) -> str:
    """Wraps a formatted magnitude in parentheses when negative."""
    return "(" + text + ")" if negative else text


def num(value: Any, decimals: int = 0) -> str:
    """Formats a monetary or count value.

    Args:
        value: Raw value, possibly None.
        decimals: Fixed decimal places.

    Returns:
        Grouped string, parenthesised when negative, or an em-dash.
    """
    if not _is_number(value):
        return DASH
    return _wrap_negative(_group(abs(float(value)), decimals), float(value) < 0)


def pct(value: Any, decimals: int = 1) -> str:
    """Formats a percentage already expressed in percentage points."""
    if not _is_number(value):
        return DASH
    return _wrap_negative(_group(abs(float(value)), decimals) + "%", float(value) < 0)


def signed_pct(value: Any, decimals: int = 1) -> str:
    """Formats a growth rate with an explicit sign.

    Growth needs its direction stated rather than implied, so this keeps the
    sign instead of using the parenthesis convention.
    """
    if not _is_number(value):
        return DASH
    return "{:+,.{d}f}%".format(float(value), d=decimals)


def mult(value: Any, decimals: int = 2) -> str:
    """Formats a ratio as a multiple, e.g. 0.23x."""
    if not _is_number(value):
        return DASH
    return _wrap_negative(_group(abs(float(value)), decimals) + "x", float(value) < 0)


def per_share(value: Any) -> str:
    """Formats a per-share figure at two decimals."""
    return num(value, 2)


def days(value: Any) -> str:
    """Formats a days-based working-capital metric."""
    if not _is_number(value):
        return DASH
    return _group(float(value), 1)


def ratio(value: Any, decimals: int = 2) -> str:
    """Formats a bare ratio with no unit suffix."""
    return num(value, decimals)


def growth(current: Any, prior: Any) -> float | None:
    """Computes period-on-period growth in percentage points.

    The collector's `_comments` growth annotations are unreliable — for a
    series' first period they restate the absolute value as a percentage —
    so growth is always derived here from the series itself.

    Args:
        current: Later-period value.
        prior: Earlier-period value.

    Returns:
        Growth in percentage points, or None when it cannot be computed. A
        non-positive base returns None rather than a misleading figure.
    """
    if not _is_number(current) or not _is_number(prior):
        return None
    if float(prior) <= 0:
        return None
    return (float(current) - float(prior)) / float(prior) * 100.0


def margin(numerator: Any, denominator: Any) -> float | None:
    """Computes a margin in percentage points, or None if not computable."""
    if not _is_number(numerator) or not _is_number(denominator):
        return None
    if float(denominator) == 0:
        return None
    return float(numerator) / float(denominator) * 100.0


def safe_div(numerator: Any, denominator: Any) -> float | None:
    """Divides two values, returning None on missing or zero denominator."""
    if not _is_number(numerator) or not _is_number(denominator):
        return None
    if float(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def pos_div(numerator: Any, denominator: Any) -> float | None:
    """Divides only when the denominator is strictly positive.

    This is the house rule for any ratio whose denominator can legitimately
    go negative: equity, EBITDA, EBIT, pre-tax profit, profit after tax,
    invested capital, capital employed, cost of goods sold. Each of those
    produces an arithmetically valid quotient against a negative base and a
    meaningless one. Negative equity divided into a negative profit yields a
    POSITIVE return on equity, which reads as strength on the page, and a
    typeset PDF lends that false authority.

    The numerator's sign is never restricted, because a negative numerator
    over a positive base is real information.

    Args:
        numerator: Value on top, any sign.
        denominator: Value underneath; must be strictly positive.

    Returns:
        The quotient, or None when either input is missing or the
        denominator is zero or negative.
    """
    if not _is_number(numerator) or not _is_number(denominator):
        return None
    if float(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)


def pos_margin(numerator: Any, denominator: Any) -> float | None:
    """Computes a percentage, only against a strictly positive base.

    The percentage-point counterpart of `pos_div`; see that function for
    why the guard is one-sided.
    """
    quotient = pos_div(numerator, denominator)
    return None if quotient is None else quotient * 100.0


def period_label(raw: str) -> str:
    """Shortens a collector period label for a table header.

    'FY 2026' becomes 'FY26' and 'JUN 2026' becomes 'Jun-26', which keeps
    column headers narrow enough for an eight-period table on A4.
    """
    text = (raw or "").strip()
    if not text:
        return DASH
    if text.upper() == "TTM":
        return "TTM"
    parts = text.split()
    if len(parts) != 2:
        return text
    head, year = parts[0], parts[1]
    suffix = year[-2:] if len(year) == 4 else year
    if head.upper() == "FY":
        return "FY" + suffix
    return head.capitalize() + "-" + suffix


def escape_typst(text: Any) -> str:
    """Escapes text for safe interpolation into Typst markup.

    Typst treats several ASCII characters as markup. Company descriptions
    and news headlines routinely contain them, so any untrusted string
    bound for a template must pass through here.
    """
    if text is None:
        return ""
    out = str(text)
    for char in ("\\", "#", "$", "*", "_", "`", "<", ">", "@", "~"):
        out = out.replace(char, "\\" + char)
    return out.replace("\r", " ").replace("\n", " ")
