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

from dataclasses import dataclass
import logging
from typing import List, Optional

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


def _div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Divides, returning None on missing or zero denominator."""
    return fmt.safe_div(numerator, denominator)


def _div_positive(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
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
    raw_material_ratio: Optional[float] = None
    sga_ratio: Optional[float] = None
    depreciation_ratio: Optional[float] = None
    ebitda_margin: Optional[float] = None
    ebit_margin: Optional[float] = None
    pat_margin: Optional[float] = None
    effective_tax_rate: Optional[float] = None

    # Returns
    roe: Optional[float] = None
    roce: Optional[float] = None
    roic: Optional[float] = None
    nopat: Optional[float] = None
    invested_capital: Optional[float] = None

    # Leverage and coverage
    debt_to_equity: Optional[float] = None
    net_debt_to_ebitda: Optional[float] = None
    interest_coverage: Optional[float] = None
    short_term_debt_share: Optional[float] = None
    goodwill_to_equity: Optional[float] = None

    # Working capital
    dso: Optional[float] = None
    dio: Optional[float] = None
    dpo: Optional[float] = None
    cash_conversion_cycle: Optional[float] = None

    # Earnings quality
    cfo_to_pat: Optional[float] = None
    accrual_ratio: Optional[float] = None
    fcf_margin: Optional[float] = None
    capex_intensity: Optional[float] = None
    capex_to_depreciation: Optional[float] = None
    retained_fcf: Optional[float] = None

    # Per share
    shares_cr: Optional[float] = None
    eps: Optional[float] = None
    dps: Optional[float] = None
    payout_pct: Optional[float] = None
    book_value_per_share: Optional[float] = None


@dataclass
class RollingPoint:
    """One trailing-four-quarter aggregate."""

    label: str
    revenue: Optional[float] = None
    ebit: Optional[float] = None
    pat: Optional[float] = None

    @property
    def ebit_margin(self) -> Optional[float]:
        """Trailing EBIT margin."""
        return fmt.margin(self.ebit, self.revenue)

    @property
    def pat_margin(self) -> Optional[float]:
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

    market_cap: Optional[float] = None
    total_debt: Optional[float] = None
    cash: Optional[float] = None
    minority_interest: Optional[float] = None
    enterprise_value: Optional[float] = None

    ev_to_ebitda: Optional[float] = None
    ev_to_ebit: Optional[float] = None
    ev_to_sales: Optional[float] = None
    ev_to_fcf: Optional[float] = None
    fcf_yield: Optional[float] = None
    earnings_yield: Optional[float] = None


@dataclass
class PerShareBridge:
    """Decomposition of earnings-per-share growth.

    EPS can rise because profit rose or because the share count fell, and
    those are different qualities of growth. Splitting them is a two-line
    calculation that no endpoint provides.
    """

    from_period: str = ""
    to_period: str = ""
    eps_start: Optional[float] = None
    eps_end: Optional[float] = None
    eps_change: Optional[float] = None
    profit_effect: Optional[float] = None
    share_count_effect: Optional[float] = None
    shares_start: Optional[float] = None
    shares_end: Optional[float] = None
    shares_retired: Optional[float] = None
    share_change_pct: Optional[float] = None


@dataclass
class DerivedAnalytics:
    """Everything this module produces for one company."""

    annual: List[AnnualMetrics]
    rolling: List[RollingPoint]
    enterprise: EnterpriseValue
    per_share: PerShareBridge
    notes: List[str]


def _annual_metrics(snap: CompanySnapshot) -> List[AnnualMetrics]:
    """Builds the per-year derived series.

    Balance-sheet and cash-flow rows are matched to income periods by their
    period label rather than by position, because the three statements can
    carry different numbers of periods.
    """
    balance_by_period = {b.period: b for b in snap.balance}
    cash_by_period = {c.period: c for c in snap.cashflow}
    prior_balance: Optional[object] = None
    out: List[AnnualMetrics] = []

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
                balance.goodwill_intangibles, balance.equity)
            row.short_term_debt_share = fmt.pos_margin(
                balance.short_term_debt, balance.debt)

            row.roe = fmt.pos_margin(income.pat, balance.equity)
            if balance.total_assets is not None and balance.current_liabilities is not None:
                capital_employed = balance.total_assets - balance.current_liabilities
                row.roce = fmt.pos_margin(income.ebit, capital_employed)

            tax_rate = income.effective_tax_rate
            if income.ebit is not None and tax_rate is not None:
                row.nopat = income.ebit * (1.0 - tax_rate / 100.0)
            if (balance.equity is not None and balance.debt is not None
                    and balance.cash is not None):
                row.invested_capital = balance.equity + balance.debt - balance.cash
            row.roic = fmt.pos_margin(row.nopat, row.invested_capital)

            row.net_debt_to_ebitda = _div_positive(balance.net_debt, income.ebitda)
            row.interest_coverage = _div_positive(income.ebit, income.interest)

            row.dso = _scaled_days(balance.receivables, income.revenue)
            if (income.raw_material_ratio is not None
                    and income.raw_material_ratio >= MIN_RAW_MATERIAL_SHARE_PCT):
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
            if (income.pat is not None and cash.cfo is not None
                    and balance is not None and prior_balance is not None
                    and balance.total_assets is not None
                    and getattr(prior_balance, "total_assets", None) is not None):
                average_assets = (balance.total_assets
                                  + prior_balance.total_assets) / 2.0
                row.accrual_ratio = fmt.pos_margin(income.pat - cash.cfo, average_assets)

        out.append(row)
        if balance is not None and income is not snap.ttm:
            prior_balance = balance

    return out


def _scaled_days(stock: Optional[float], flow: Optional[float]) -> Optional[float]:
    """Converts a balance against an annual flow into a day count."""
    ratio = _div_positive(stock, flow)
    return None if ratio is None else ratio * DAYS_IN_YEAR


def _rolling(snap: CompanySnapshot, window: int = 4) -> List[RollingPoint]:
    """Builds trailing-four-quarter aggregates from the quarterly series.

    A single TTM figure is one dot. Rolling the window across the reported
    quarters produces a seasonality-free trend line, which is what makes a
    quarterly series readable for a business with an uneven year.
    """
    quarters = [q for q in snap.quarters if q.revenue is not None]
    if len(quarters) < window:
        return []

    out: List[RollingPoint] = []
    for end in range(window - 1, len(quarters)):
        block = quarters[end - window + 1:end + 1]

        def total(attribute: str) -> Optional[float]:
            values = [getattr(q, attribute) for q in block]
            if any(v is None for v in values):
                return None
            return sum(values)

        out.append(RollingPoint(
            label=fmt.period_label(block[-1].period),
            revenue=total("revenue"),
            ebit=total("ebit"),
            pat=total("pat"),
        ))
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
    notes: List[str] = []
    annual = _annual_metrics(snap)

    # A leverage table of nothing but em-dashes is noise. Banks report no
    # consolidated total-debt line — deposits and borrowings are separate —
    # so the whole family collapses and the exhibit is better omitted with
    # an explanation than printed empty.
    if not any(
        v is not None
        for row in annual
        for v in (row.debt_to_equity, row.net_debt_to_ebitda,
                  row.interest_coverage, row.short_term_debt_share,
                  row.goodwill_to_equity)
    ):
        notes.append(
            "Leverage and coverage ratios are unavailable: the provider reports "
            "no consolidated total-debt line for this company, which is normal "
            "for a bank, where deposits and borrowings are disclosed separately."
        )

    latest = annual[-1] if annual else None
    if latest is not None and latest.cash_conversion_cycle is None:
        if (latest.raw_material_ratio is not None
                and latest.raw_material_ratio < MIN_RAW_MATERIAL_SHARE_PCT):
            notes.append(
                "Inventory and payable days are withheld: reported raw-material "
                "cost is %.1f%% of revenue, too small a base to yield a "
                "meaningful day count for a services business."
                % latest.raw_material_ratio
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
