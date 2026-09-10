# Spec: Reporting — the Quantitative Engine

- **Status**: As-built reference, growing
- **Author**: Claude, for @jenish.gajera
- **Date**: 2026-09-10
- **Covers**: `reporting/` — currently `fmt.py`. Extended by
  [`code-style-refactor.md`](../plans/code-style-refactor.md) Phase 3 with `composites.py`,
  `selfcheck.py`, `snapshot.py`, `analytics.py` and `charts.py`.
- **Guarded by**: `scripts/verify_reporting.py`

Receiving document for design rationale relocated out of `reporting/` docstrings. Created in
Phase 2 because `fmt.py` needed it; F6 scheduled it for Phase 3, which will fill in the rest.

Nothing here is new — it is the same reasoning, moved so the code reads as code.

---

## 1. Typesetting conventions (`fmt.py`)

Every figure that reaches a page passes through this module. Centralising it is what makes
precision consistent: ragged decimals across a table are the clearest signal that a financial
document was generated rather than designed.

- **International digit grouping** (1,234,567), never lakh-crore grouping. Both reference broker
  notes use this, and mixing the two within one document is worse than either choice.
- **Units are declared in column headers**, never repeated per cell.
- **Negatives render in parentheses** — the long-standing finance convention. A bare minus sign
  is easy to miss at 7pt.
- **Absent data renders as an em-dash**, which must stay visually distinct from a real zero.
  **A rendered `0.00` for missing data is a factual error**, not a cosmetic one.

That last rule is the zero-hallucination rule applied to typography: the page must never assert
a number the source did not contain.

---

## 2. Why `pos_div` refuses a non-positive denominator

`pos_div` is the house rule for any ratio whose denominator can legitimately go negative:
equity, EBITDA, EBIT, pre-tax profit, profit after tax, invested capital, capital employed, cost
of goods sold.

Each of those produces an arithmetically valid quotient against a negative base and a
**meaningless** one. The motivating case:

> Negative equity divided into a negative profit yields a **positive** return on equity.

That reads as strength on the page, and a typeset PDF lends it false authority. A distressed
company with negative net worth would show a healthy ROE. So the denominator must be strictly
positive or the result is `None`, which `fmt.py` renders as an em-dash per §1.

**The numerator's sign is never restricted** — a negative numerator over a positive base is real
information, and suppressing it would hide a genuine loss.

`pos_ratio` applies the same rule and returns a percentage.

Guarded by `check_division_helpers` in `scripts/verify_reporting.py`, and the constants are in
`verify_style.py`'s frozen set because the sign semantics are a contract, not a preference.
