"""Derived analytics computed from a CompanySnapshot.

The collector's ratio endpoints each return a single trailing-twelve-month
point. The statement endpoints return ten years and twelve quarters. This
module turns the former into the latter: the same ratios as time series,
which is where the analysis actually lives — a level tells you where a
company is, a series tells you which way it is going and how steadily.

Methodology is verified against the provider rather than assumed. Every
formula here reproduces the corresponding endpoint's own TTM figure exactly
for Wipro — ROIC 30.81%, ROE 14.98%, interest coverage 14.03x, net
debt/EBITDA -1.36x, DSO 50.2 days — so the trailing column of each series
agrees with the endpoint it extends.

Two conventions worth stating:

    - Returns use PERIOD-END equity and invested capital, not two-point
      averages, because that is what the provider's DuPont and
      capital-efficiency endpoints do. Matching them matters more than
      textbook preference; a report that disagrees with its own source
      data is worse than one using a slightly cruder denominator.
    - Any ratio with a non-positive or missing denominator returns None,
      never a number. Negative equity and negative EBITDA both produce
      arithmetically valid but analytically meaningless ratios, and a
      typeset PDF lends them false authority.
"""

import logging
from dataclasses import dataclass

from reporting import fmt
from reporting.snapshot import CompanySnapshot

logger = logging.getLogger(__name__)

DAYS_IN_YEAR: float = 365.0

# Inventory and payable days divide by cost of goods sold. The provider
# supplies raw-material cost only, which approximates COGS for a
# manufacturer but not for a services business — 64% of revenue at
# Reliance against 0.6% at Wipro. Below this share the denominator is too
# small to produce a meaningful day count, so the cycle is withheld.
MIN_RAW_MATERIAL_SHARE_PCT: float = 15.0


def _div_positive(numerator: float | None, denominator: float | None) -> float | None:
    """Divides only when the denominator is strictly positive.

    Used for ratios whose sign becomes nonsensical against a negative base,
    such as net debt to EBITDA when EBITDA is negative.
    """
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


@dataclass
class AnnualMetrics:
    """Derived metrics for one fiscal period."""

    period: str

    # Cost structure, as a percentage of revenue
    raw_material_ratio: float | None = None
    sga_ratio: float | None = None
    depreciation_ratio: float | None = None
    ebitda_margin: float | None = None
    ebit_margin: float | None = None
    pat_margin: float | None = None
    effective_tax_rate: float | None = None

    # Returns
    roe: float | None = None
    roce: float | None = None
    roic: float | None = None
    nopat: float | None = None
    invested_capital: float | None = None

    # Leverage and coverage
    debt_to_equity: float | None = None
    net_debt_to_ebitda: float | None = None
    interest_coverage: float | None = None
    short_term_debt_share: float | None = None
    goodwill_to_equity: float | None = None

    # Working capital
    dso: float | None = None
    dio: float | None = None
    dpo: float | None = None
    cash_conversion_cycle: float | None = None

    # Earnings quality
    cfo_to_pat: float | None = None
    accrual_ratio: float | None = None
    fcf_margin: float | None = None
    capex_intensity: float | None = None
    capex_to_depreciation: float | None = None
    retained_fcf: float | None = None

    # Per share
    shares_cr: float | None = None
    eps: float | None = None
    dps: float | None = None
    payout_pct: float | None = None
    book_value_per_share: float | None = None


@dataclass
class RollingPoint:
    """One trailing-four-quarter aggregate."""

    label: str
    revenue: float | None = None
    ebit: float | None = None
    pat: float | None = None

    @property
    def ebit_margin(self) -> float | None:
        """Trailing EBIT margin."""
        return fmt.margin(self.ebit, self.revenue)

    @property
    def pat_margin(self) -> float | None:
        """Trailing PAT margin."""
        return fmt.margin(self.pat, self.revenue)


@dataclass
class EnterpriseValue:
    """Enterprise value bridge and the multiples built on it.

    The collector exposes P/E, P/B and dividend yield but no enterprise
    value and no EV multiple, despite EV/EBITDA being among the most-cited
    measures in institutional work. Minority interest is included in the
    bridge because the enterprise's earnings are consolidated whole.
    """

    market_cap: float | None = None
    total_debt: float | None = None
    cash: float | None = None
    minority_interest: float | None = None
    enterprise_value: float | None = None

    ev_to_ebitda: float | None = None
    ev_to_ebit: float | None = None
    ev_to_sales: float | None = None
    ev_to_fcf: float | None = None
    fcf_yield: float | None = None
    earnings_yield: float | None = None


@dataclass
class PerShareBridge:
    """Decomposition of earnings-per-share growth.

    EPS can rise because profit rose or because the share count fell, and
    those are different qualities of growth. Splitting them is a two-line
    calculation that no endpoint provides.
    """

    from_period: str = ""
    to_period: str = ""
    eps_start: float | None = None
    eps_end: float | None = None
    eps_change: float | None = None
    profit_effect: float | None = None
    share_count_effect: float | None = None
    shares_start: float | None = None
    shares_end: float | None = None
    shares_retired: float | None = None
    share_change_pct: float | None = None


@dataclass
class DerivedAnalytics:
    """Everything this module produces for one company."""

    annual: list[AnnualMetrics]
    rolling: list[RollingPoint]
    enterprise: EnterpriseValue
    per_share: PerShareBridge
    notes: list[str]


def _annual_metrics(snap: CompanySnapshot) -> list[AnnualMetrics]:
    """Builds the per-year derived series.

    Balance-sheet and cash-flow rows are matched to income periods by their
    period label rather than by position, because the three statements can
    carry different numbers of periods.
    """
    balance_by_period = {b.period: b for b in snap.balance}
    cash_by_period = {c.period: c for c in snap.cashflow}
    prior_balance: object | None = None
    out: list[AnnualMetrics] = []

    periods = list(snap.years) + ([snap.ttm] if snap.ttm else [])
    for income in periods:
        # The TTM income row has no balance sheet of its own; the latest
        # reported balance sheet is the correct pairing for it.
        balance = balance_by_period.get(income.period)
        if balance is None and income is snap.ttm and snap.balance:
            balance = snap.balance[-1]
        cash = cash_by_period.get(income.period)

        row = AnnualMetrics(
            period=income.period,
            raw_material_ratio=income.raw_material_ratio,
            sga_ratio=income.sga_ratio,
            depreciation_ratio=income.depreciation_ratio,
            ebitda_margin=income.ebitda_margin,
            ebit_margin=income.ebit_margin,
            pat_margin=income.pat_margin,
            effective_tax_rate=income.effective_tax_rate,
            eps=income.eps,
            dps=income.dps,
            payout_pct=income.payout_pct,
        )

        if balance is not None:
            row.shares_cr = balance.shares_cr
            row.book_value_per_share = balance.book_value_per_share
            row.debt_to_equity = balance.debt_to_equity
            row.goodwill_to_equity = fmt.pos_margin(
                balance.goodwill_intangibles, balance.equity
            )
            row.short_term_debt_share = fmt.pos_margin(
                balance.short_term_debt, balance.debt
            )

            row.roe = fmt.pos_margin(income.pat, balance.equity)
            if (
                balance.total_assets is not None
                and balance.current_liabilities is not None
            ):
                capital_employed = balance.total_assets - balance.current_liabilities
                row.roce = fmt.pos_margin(income.ebit, capital_employed)

            tax_rate = income.effective_tax_rate
            if income.ebit is not None and tax_rate is not None:
                row.nopat = income.ebit * (1.0 - tax_rate / 100.0)
            if (
                balance.equity is not None
                and balance.debt is not None
                and balance.cash is not None
            ):
                row.invested_capital = balance.equity + balance.debt - balance.cash
            row.roic = fmt.pos_margin(row.nopat, row.invested_capital)

            row.net_debt_to_ebitda = _div_positive(balance.net_debt, income.ebitda)
            row.interest_coverage = _div_positive(income.ebit, income.interest)

            row.dso = _scaled_days(balance.receivables, income.revenue)
            if (
                income.raw_material_ratio is not None
                and income.raw_material_ratio >= MIN_RAW_MATERIAL_SHARE_PCT
            ):
                row.dio = _scaled_days(balance.inventory, income.raw_materials)
                row.dpo = _scaled_days(balance.payables, income.raw_materials)
                if row.dso is not None and row.dio is not None and row.dpo is not None:
                    row.cash_conversion_cycle = row.dso + row.dio - row.dpo

        if cash is not None:
            row.cfo_to_pat = fmt.pos_margin(cash.cfo, income.pat)
            row.fcf_margin = fmt.margin(cash.fcf, income.revenue)
            row.capex_intensity = fmt.margin(cash.capex, income.revenue)
            row.capex_to_depreciation = _div_positive(cash.capex, income.depreciation)
            if cash.fcf is not None and cash.dividends_paid is not None:
                row.retained_fcf = cash.fcf - cash.dividends_paid
            # Accruals are scaled by average total assets, so the first
            # period of the series has no value.
            if (
                income.pat is not None
                and cash.cfo is not None
                and balance is not None
                and prior_balance is not None
                and balance.total_assets is not None
                and getattr(prior_balance, "total_assets", None) is not None
            ):
                average_assets = (
                    balance.total_assets + prior_balance.total_assets
                ) / 2.0
                row.accrual_ratio = fmt.pos_margin(
                    income.pat - cash.cfo, average_assets
                )

        out.append(row)
        if balance is not None and income is not snap.ttm:
            prior_balance = balance

    return out


def _scaled_days(stock: float | None, flow: float | None) -> float | None:
    """Converts a balance against an annual flow into a day count."""
    ratio = _div_positive(stock, flow)
    return None if ratio is None else ratio * DAYS_IN_YEAR


def _rolling(snap: CompanySnapshot, window: int = 4) -> list[RollingPoint]:
    """Builds trailing-four-quarter aggregates from the quarterly series.

    A single TTM figure is one dot. Rolling the window across the reported
    quarters produces a seasonality-free trend line, which is what makes a
    quarterly series readable for a business with an uneven year.
    """
    quarters = [q for q in snap.quarters if q.revenue is not None]
    if len(quarters) < window:
        return []

    out: list[RollingPoint] = []
    for end in range(window - 1, len(quarters)):
        block = quarters[end - window + 1 : end + 1]

        def total(attribute: str, block: list = block) -> float | None:
            values = [getattr(q, attribute) for q in block]
            if any(v is None for v in values):
                return None
            return sum(values)

        out.append(
            RollingPoint(
                label=fmt.period_label(block[-1].period),
                revenue=total("revenue"),
                ebit=total("ebit"),
                pat=total("pat"),
            )
        )
    return out


def _enterprise(snap: CompanySnapshot) -> EnterpriseValue:
    """Builds the enterprise-value bridge and its multiples."""
    balance = snap.latest_balance
    income = snap.ttm or (snap.years[-1] if snap.years else None)
    cash_row = snap.cashflow[-1] if snap.cashflow else None

    market_cap = snap.market_cap_cr or snap.implied_market_cap_cr
    result = EnterpriseValue(market_cap=market_cap)
    if balance is None:
        return result

    result.total_debt = balance.debt
    result.cash = balance.cash
    result.minority_interest = balance.minority_interest

    if market_cap is None or balance.debt is None or balance.cash is None:
        return result
    result.enterprise_value = (
        market_cap + balance.debt - balance.cash + (balance.minority_interest or 0.0)
    )

    if income is not None:
        result.ev_to_ebitda = _div_positive(result.enterprise_value, income.ebitda)
        result.ev_to_ebit = _div_positive(result.enterprise_value, income.ebit)
        result.ev_to_sales = _div_positive(result.enterprise_value, income.revenue)
        result.earnings_yield = fmt.margin(income.pat, market_cap)
    if cash_row is not None:
        result.ev_to_fcf = _div_positive(result.enterprise_value, cash_row.fcf)
        result.fcf_yield = fmt.margin(cash_row.fcf, market_cap)
    return result


def _per_share(snap: CompanySnapshot) -> PerShareBridge:
    """Decomposes EPS change into profit and share-count effects."""
    years = [y for y in snap.years if y.eps is not None and y.pat is not None]
    balance_by_period = {b.period: b for b in snap.balance}
    usable = [(y, balance_by_period.get(y.period)) for y in years]
    usable = [(y, b) for y, b in usable if b is not None and b.shares_cr]
    if len(usable) < 2:
        return PerShareBridge()

    (start_income, start_balance) = usable[0]
    (end_income, end_balance) = usable[-1]
    shares_start = start_balance.shares_cr
    shares_end = end_balance.shares_cr

    bridge = PerShareBridge(
        from_period=fmt.period_label(start_income.period),
        to_period=fmt.period_label(end_income.period),
        eps_start=start_income.eps,
        eps_end=end_income.eps,
        shares_start=shares_start,
        shares_end=shares_end,
    )
    bridge.eps_change = end_income.eps - start_income.eps
    bridge.shares_retired = shares_start - shares_end
    bridge.share_change_pct = fmt.margin(shares_end - shares_start, shares_start)

    # Holding the share count fixed isolates the profit contribution; the
    # residual is what the change in share count did.
    bridge.profit_effect = (end_income.pat - start_income.pat) / shares_start
    bridge.share_count_effect = bridge.eps_change - bridge.profit_effect
    return bridge


def compute(snap: CompanySnapshot) -> DerivedAnalytics:
    """Computes every Tier 1 derived metric for a company.

    Args:
        snap: Normalised company snapshot.

    Returns:
        DerivedAnalytics, with `notes` recording any metric family that had
        to be withheld so the report can say so rather than omit silently.

    """
    notes: list[str] = []
    annual = _annual_metrics(snap)

    # A leverage table of nothing but em-dashes is noise. Banks report no
    # consolidated total-debt line — deposits and borrowings are separate —
    # so the whole family collapses and the exhibit is better omitted with
    # an explanation than printed empty.
    if not any(
        v is not None
        for row in annual
        for v in (
            row.debt_to_equity,
            row.net_debt_to_ebitda,
            row.interest_coverage,
            row.short_term_debt_share,
            row.goodwill_to_equity,
        )
    ):
        notes.append(
            "Leverage and coverage ratios are unavailable: the provider reports "
            "no consolidated total-debt line for this company, which is normal "
            "for a bank, where deposits and borrowings are disclosed separately."
        )

    latest = annual[-1] if annual else None
    if latest is not None and latest.cash_conversion_cycle is None:
        if (
            latest.raw_material_ratio is not None
            and latest.raw_material_ratio < MIN_RAW_MATERIAL_SHARE_PCT
        ):
            notes.append(
                "Inventory and payable days are withheld: reported raw-material "
                f"cost is {latest.raw_material_ratio:.1f}% of revenue, too small a base "
                "to yield a "
                "meaningful day count for a services business."
            )
        elif latest.dso is None:
            notes.append(
                "Working-capital days are unavailable: the provider reports no "
                "receivables, inventory or payables for this company."
            )

    return DerivedAnalytics(
        annual=annual,
        rolling=_rolling(snap),
        enterprise=_enterprise(snap),
        per_share=_per_share(snap),
        notes=notes,
    )
