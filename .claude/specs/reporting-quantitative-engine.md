# Spec: Reporting — the Quantitative Engine

- **Status**: As-built reference, growing
- **Author**: Claude, for @jenish.gajera
- **Date**: 2026-09-10
- **Covers**: all of `reporting/` — `fmt.py`, `snapshot.py`, `analytics.py`, `composites.py`,
  `selfcheck.py`, `charts.py`, `tokens.py`, `typst_doc.py`.
- **Guarded by**: `scripts/verify_reporting.py`

Receiving document for design rationale relocated out of `reporting/` docstrings. Created in
Phase 2 for `fmt.py`; completed in Phase 3.

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

---

## 3. `snapshot.py` — the collector quirks, absorbed once

The report templates read only this model, never raw JSON. That boundary is where the unit
contract is enforced and where five upstream quirks are absorbed once instead of in every table:

- **`income/annual` appends a `TTM` row that is not a fiscal year.** Split out rather than
  plotted alongside FY columns.
- **Series arrive oldest-first and of varying length** (15 quarters, 11 annual rows), so windows
  are taken from the tail. `QUARTERS_SHOWN` / `YEARS_SHOWN` cap what is exposed.
- **`balTcso` is share count in crore**, cross-checked against PAT / EPS.
- **Peer `marketCap` is denominated in rupee MILLION** while every other figure is rupee CRORE.
  Converted on ingest; see `MN_PER_CR = 10`. This is the easiest unit error in the codebase to
  reintroduce.
- **The `_comments` growth annotations are unusable** — for a series' first period they restate
  the absolute value as a percentage — so growth is always derived (`fmt.growth`).

---

## 4. `analytics.py` — Tier 1, and the calibration that proves it

The collector's ratio endpoints each return a single trailing-twelve-month point; the statement
endpoints return ten years and twelve quarters. This module turns the former into the latter: the
same ratios as time series, which is where the analysis lives. A level says where a company is;
a series says which way it is going and how steadily.

**Methodology is verified against the provider, not assumed.** Every formula here reproduces the
corresponding endpoint's own TTM figure exactly for Wipro:

| Metric | Provider TTM |
| :--- | ---: |
| ROIC | 30.81% |
| ROE | 14.98% |
| Interest coverage | 14.03x |
| Net debt / EBITDA | -1.36x |
| DSO | 50.2 days |

So the trailing column of each series agrees with the endpoint it extends. **These are
calibration values: a change that moves any of them is a methodology change, not a refactor.**

Two conventions:

- **Returns use period-end equity and invested capital**, not two-point averages, because that is
  what the provider's DuPont and capital-efficiency endpoints do. Matching the source matters
  more than textbook preference — a report that disagrees with its own data is worse than one
  using a slightly cruder denominator.
- **Any ratio with a non-positive or missing denominator returns `None`**, never a number. Same
  rule as section 2, enforced through `_div_positive`.

---

## 5. `composites.py` — Tier 2, and one architectural rule

> **A composite is never reported as a bare number.**

"F-Score 7" is not analysis. The nine sub-tests *are* the analysis; the total is a convenience
for sorting. So every composite carries its own components as data — `Piotroski.tests` holds nine
`ScoreTest` records with inputs, threshold and verdict; `Altman.components` holds five, each with
its ratio, published weight and resulting contribution. **The score is a derived property of a
structure that contains its parts**, so a renderer cannot print a total without holding the parts.

Guarded by `check_composites_carry_components`, `check_piotroski_is_nine_binary_tests` and
`check_altman_score_is_sum_of_terms`.

### Three deliberate departures from the textbook

Each is surfaced in the report rather than buried in the code.

1. **Piotroski's eighth signal needs gross margin, which this provider cannot support.**
   `incGpro` is null for every ticker tested, and `incRaw` carries raw materials only — 64% of
   revenue at Reliance, 0.6% at Wipro, so it is not a usable proxy. EBITDA margin is substituted
   and recorded in `Piotroski.substitutions` so the page can declare it.
2. **Piotroski scales profitability by *opening* total assets**, as the 2000 paper specifies.
   That differs from the period-end convention section 4 uses to match the provider. The
   framework's definition wins inside the framework's own exhibit, and every test states the base
   it used.
3. **Altman and the reinvestment identity are withheld entirely for financials.** Working
   capital, and the ratio of sales to assets, do not describe a bank; a Z-Score computed from a
   deposit book is a number with no meaning attached. Detection is two independent signals,
   either sufficient: the provider's sector label against `FINANCIAL_SECTOR_MARKERS`, and the
   absence of a consolidated total-debt line — which catches a financial whose sector string is
   unhelpful, because a bank discloses deposits and borrowings separately and never a single debt
   figure. Guarded by `check_financial_sector_withholds_altman`.

---

## 6. `selfcheck.py` — three kinds of check, in descending severity

Every figure in a report is either something the provider said or something this codebase
computed. A typeset PDF gives a wrong number the same authority as a right one, so this module
re-derives what can be re-derived by an independent route and reports where the two disagree.

- **Identities** must hold to floating-point precision, because they are definitions rather than
  estimates. The five DuPont factors multiplied together *are* ROE. Operating plus investing plus
  financing cash flow *is* the movement in cash. Sources equal uses. **A failure here is a defect
  in this codebase, not a data problem**, and is reported as such. `TOL_IDENTITY_PCT`.
- **Reconciliations** compare a figure computed here against the provider's published value for
  the same quantity, expected to agree within rounding: the provider publishes DuPont factors to
  four decimals, so a rebuilt product lands a few thousandths of a percentage point from its
  stated ROE. A wide gap means a methodology difference worth knowing about. `TOL_PROVIDER_PCT`.
- **Guardrails** assert the report's editorial rules held *in practice* — no ratio published
  against a non-positive denominator, no composite reaching the page without its components.
  These cannot be proven from the code alone, because whether a path was taken depends on the
  company's numbers. `TOL_METHODOLOGY_PCT`.

The tolerance bands must stay ordered identity < provider < methodology
(`check_selfcheck_tolerances`), and the suite must keep all 20 checks
(`check_selfcheck_suite_is_intact`).

**The result is rendered as an exhibit, not kept in a log.** A report that states its own
arithmetic was verified, and shows the residuals, is making a checkable claim; one that stays
silent is asking for trust.

---

## 7. `charts.py` and `tokens.py` — house style

SVG so Typst embeds vector artwork that stays sharp in print. Text is converted to paths on save,
making output independent of the fonts installed on the rendering host.

- **Charts carry no title.** The Typst exhibit caption titles them, so the document's own typeface
  and hierarchy apply. A chart title in a different face is the clearest sign a figure was pasted
  into a report rather than designed with it.
- **Bars always start at zero.** A truncated bar axis misstates magnitude.
- Horizontal hairline gridlines only, behind the data, no left spine — the gridlines carry the
  scale.
- The latest period is emphasised, since that is what a reader looks for first.
- **No label may overlap another.** Where a scatter would collide, the chart *form* changes rather
  than the labels shrinking.
- Every chart returns `None` rather than raising when the data cannot support it
  (`check_charts_withhold_rather_than_raise` covers all 15).

**Colour policy**, in order of importance:

1. **`BRAND` is never a data colour.** It marks masthead, section rules and page furniture only.
   The moment it encodes a series, the reader loses the ability to distinguish branding from
   meaning. Guarded by `check_design_tokens_are_valid`.
2. `POSITIVE` and `NEGATIVE` encode direction and nothing else.
3. The series ramp steps **lightness rather than hue**, so ordering survives greyscale printing.

**Typeface policy: every face must have lining figures.** Georgia and Constantia were both
rejected despite being good print serifs, because they default to old-style (text) figures where
digits sit at varying heights — in a column of financials that reads as broken. Cambria and
Segoe UI ship lining figures and support tabular widths.

---

## 8. `typst_doc.py` — document structure

Tables are emitted as native Typst markup rather than through an intermediate format. Typst's
`table` gives column spanners, per-column alignment and explicit stroke control directly, so
there is nothing to gain from generating HTML or LaTeX first and everything to lose in fidelity.

Conventions taken from how sell-side research is actually set:

- **Every table and chart is a numbered exhibit** with a caption above and a source line below.
  Numbering is what lets body text refer to a figure, and is the clearest marker of
  research-grade structure.
- **Sections are semantic, not incidental**: each answers one question, and an exhibit appears
  only in the section whose question it addresses. The margin table sits with the income
  statement it decomposes; the leverage series with the balance sheet it is drawn from; the cash
  cycle with earnings quality, because both ask whether reported profit is real.
- Content is paired two-up wherever both halves are narrow, so a page fills instead of leaving a
  column of white beside a short table.
- **Booktabs discipline**: no vertical rules, three horizontal ones — above the header, below the
  header, below the body.
