"""Tier 2 composites: scored frameworks and identity-based decompositions.

Tier 1 (see `analytics`) turns the provider's single trailing ratios into
series. This module builds the composites that sit on top of those series:
two published scoring frameworks, two cash-allocation identities, and the
DuPont decomposition extended across every reported year.

One rule governs every exhibit here, and it is the reason the module is
shaped the way it is:

    A COMPOSITE IS NEVER REPORTED AS A BARE NUMBER.

"F-Score 7" is not analysis. The nine sub-tests are the analysis; the total
is a convenience for sorting. So every composite here carries its own
components as data - `Piotroski.tests` holds nine `ScoreTest` records with
the inputs, the threshold and the verdict for each; `Altman.components`
holds five, each with its ratio, its published weight and the contribution
that weight produced. The score is a derived PROPERTY of a structure that
contains its parts, so the renderer cannot print a total without having the
parts in hand.

The second rule is inherited from Tier 1 and enforced through
`fmt.pos_div` / `fmt.pos_margin`: no ratio is computed against a
non-positive denominator. Total assets, total liabilities, equity, revenue,
EBIT, pre-tax profit and NOPAT can all arrive at or below zero, and each
would otherwise yield an arithmetically clean, analytically empty figure.

Three deliberate departures from the textbook, each surfaced in the report
rather than buried here:

    - Piotroski's eighth signal needs GROSS margin, which this provider
      cannot support: `incGpro` is null for every ticker tested and
      `incRaw` carries raw materials only (64% of revenue at Reliance,
      0.6% at Wipro). EBITDA margin is substituted, and the substitution is
      recorded in `Piotroski.substitutions` so the page can declare it.
    - Piotroski scales profitability by OPENING total assets, as the 2000
      paper specifies. That differs from the period-end convention the
      returns section uses to match the provider's own endpoints. The
      framework's definition wins inside the framework's own exhibit, and
      every test states the base it used.
    - Altman and the reinvestment identity are withheld entirely for
      financials. Working capital, and the ratio of sales to assets, do not
      describe a bank; a Z-Score computed from a deposit book is a number
      with no meaning attached to it.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Sequence, Tuple

from reporting import fmt, tokens
from reporting.analytics import AnnualMetrics, DerivedAnalytics
from reporting.snapshot import BalancePeriod, CashflowPeriod, CompanySnapshot, IncomePeriod

logger = logging.getLogger(__name__)

# Sector strings that mark a lender or insurer. Matched case-insensitively
# as substrings against the provider's sector label, which arrives as free
# text such as 'Private Banks' or 'Insurance'.
FINANCIAL_SECTOR_MARKERS: Tuple[str, ...] = (
    "bank", "financ", "nbfc", "insur", "capital market",
    "asset management", "broker", "lending", "housing finance",
)

# Piotroski's share-issuance signal asks whether the company raised equity.
# Only the share COUNT is available here, and a count drifts upward every
# year on employee-option vesting at companies that have raised nothing.
# Growth below this threshold is read as that drift rather than a raise.
SHARE_ISSUE_TOLERANCE_PCT: float = 0.5

# Altman (1968) coefficients and cut-offs, the listed-company model.
ALTMAN_PUBLIC_WEIGHTS: Dict[str, float] = {
    "working_capital": 1.2,
    "retained_earnings": 1.4,
    "ebit": 3.3,
    "equity_value": 0.6,
    "sales": 1.0,
}
ALTMAN_PUBLIC_SAFE: float = 2.99
ALTMAN_PUBLIC_DISTRESS: float = 1.81

# Altman (1983) Z-prime, the private-company revision, used for the
# historical series because no share price is available for past years and
# the equity term therefore has to be taken at book. Its coefficients are
# not interchangeable with the listed model's.
ALTMAN_PRIVATE_WEIGHTS: Dict[str, float] = {
    "working_capital": 0.717,
    "retained_earnings": 0.847,
    "ebit": 3.107,
    "equity_value": 0.420,
    "sales": 0.998,
}
ALTMAN_PRIVATE_SAFE: float = 2.90
ALTMAN_PRIVATE_DISTRESS: float = 1.23


# --- shared value types ---------------------------------------------------


@dataclass
class ScoreTest:
    """One binary sub-test of a scored framework.

    Carries everything the page needs in order to show WHY a point was or
    was not awarded: what was tested, the figure this period, the figure it
    was measured against, and the verdict.

    Every one of Piotroski's nine signals has the same shape - a value
    against a comparator - which is why one record type covers all of them.
    The comparator is the prior year for the six change tests, zero for the
    two sign tests, and the same year's profit for the accrual test.
    """

    number: int
    name: str
    definition: str
    value: Optional[float] = None
    comparator: Optional[float] = None
    unit: str = "pct"
    passed: Optional[bool] = None
    unavailable_reason: str = ""

    @property
    def points(self) -> Optional[int]:
        """One when the test passed, zero when it failed, None if untested."""
        if self.passed is None:
            return None
        return 1 if self.passed else 0

    @property
    def computable(self) -> bool:
        """Whether the test reached a verdict."""
        return self.passed is not None


@dataclass
class Piotroski:
    """The nine-signal F-Score for one period, with its sub-tests."""

    period: str = ""
    prior_period: str = ""
    tests: List[ScoreTest] = field(default_factory=list)
    substitutions: List[str] = field(default_factory=list)
    withheld: List[str] = field(default_factory=list)

    @property
    def score(self) -> Optional[int]:
        """Points awarded across the sub-tests that reached a verdict."""
        awarded = [t.points for t in self.tests if t.points is not None]
        return sum(awarded) if awarded else None

    @property
    def computable(self) -> int:
        """How many of the nine signals could be evaluated."""
        return sum(1 for t in self.tests if t.computable)

    @property
    def comparable(self) -> bool:
        """Whether all nine signals were evaluated.

        False means the total is not a standard F-Score and must not be
        compared against one, however tempting the single digit looks.
        """
        return self.computable == 9

    @property
    def profitability_points(self) -> Optional[int]:
        """Points from signals 1-4, the profitability group."""
        return _group_points(self.tests, 1, 4)

    @property
    def leverage_points(self) -> Optional[int]:
        """Points from signals 5-7, the leverage and liquidity group."""
        return _group_points(self.tests, 5, 7)

    @property
    def efficiency_points(self) -> Optional[int]:
        """Points from signals 8-9, the operating-efficiency group."""
        return _group_points(self.tests, 8, 9)


@dataclass
class PiotroskiPoint:
    """One period of the F-Score series."""

    period: str
    score: Optional[int] = None
    computable: int = 0


@dataclass
class AltmanComponent:
    """One weighted term of a Z-Score."""

    key: str
    name: str
    definition: str
    ratio: Optional[float] = None
    weight: float = 0.0

    @property
    def contribution(self) -> Optional[float]:
        """The term's contribution to the score."""
        return None if self.ratio is None else self.ratio * self.weight


@dataclass
class Altman:
    """A Z-Score with the five terms that produced it."""

    period: str = ""
    model: str = ""
    equity_basis: str = ""
    components: List[AltmanComponent] = field(default_factory=list)
    safe_above: float = ALTMAN_PUBLIC_SAFE
    distress_below: float = ALTMAN_PUBLIC_DISTRESS
    withheld_reason: str = ""

    @property
    def score(self) -> Optional[float]:
        """Sum of the weighted terms, or None if any term is missing.

        A Z-Score built from four of five terms is not a low Z-Score, it is
        no Z-Score, so a missing term withholds the total rather than
        quietly shrinking it.
        """
        if not self.components:
            return None
        contributions = [c.contribution for c in self.components]
        if any(c is None for c in contributions):
            return None
        return sum(contributions)

    @property
    def zone(self) -> str:
        """The published band the score falls in."""
        score = self.score
        if score is None:
            return ""
        if score >= self.safe_above:
            return "Safe"
        if score <= self.distress_below:
            return "Distress"
        return "Grey"

    @property
    def dominant(self) -> Optional[AltmanComponent]:
        """The term supplying more than half the score, if one does.

        Altman fitted this model on manufacturers, and the equity term
        misbehaves outside that setting: an asset-light company with a large
        market capitalisation and almost no liabilities drives the fourth
        term alone into double digits, so the total reads as a solvency
        verdict when it is mostly a valuation observation. TCS scores 11.0
        with 6.9 of it from that one term. Naming the dominant term is what
        keeps the total from being read as something it is not.
        """
        score = self.score
        if score is None or score <= 0:
            return None
        for component in self.components:
            contribution = component.contribution
            if contribution is not None and contribution > score / 2.0:
                return component
        return None

    @property
    def concentration_pct(self) -> Optional[float]:
        """Share of the score contributed by its largest single term."""
        component = self.dominant
        if component is None:
            return None
        return fmt.pos_margin(component.contribution, self.score)


@dataclass
class AltmanPoint:
    """One period of the Z-prime series."""

    period: str
    score: Optional[float] = None
    zone: str = ""


@dataclass
class FlowItem:
    """One line of a sources-and-uses statement."""

    label: str
    amount: Optional[float] = None
    definition: str = ""

    @property
    def magnitude(self) -> Optional[float]:
        """Absolute size, which is how a sources-and-uses table reads."""
        return None if self.amount is None else abs(self.amount)


@dataclass
class SourcesAndUses:
    """Cumulative capital allocation over the reported window.

    Built on the cash-flow statement's own articulation - operating plus
    investing plus financing equals the change in cash - which means the two
    columns balance by construction rather than by assertion. That identity
    was verified against all forty reported periods of the four tickers
    tested before this exhibit was built on it, and `selfcheck` re-verifies
    it for every report produced.
    """

    from_period: str = ""
    to_period: str = ""
    years: int = 0
    sources: List[FlowItem] = field(default_factory=list)
    uses: List[FlowItem] = field(default_factory=list)
    debt_change: Optional[float] = None
    share_change_pct: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    @property
    def total_sources(self) -> Optional[float]:
        """Sum of the sources column."""
        return _sum_magnitudes(self.sources)

    @property
    def total_uses(self) -> Optional[float]:
        """Sum of the uses column."""
        return _sum_magnitudes(self.uses)

    @property
    def residual(self) -> Optional[float]:
        """Sources less uses. Must be zero; `selfcheck` asserts that."""
        sources, uses = self.total_sources, self.total_uses
        if sources is None or uses is None:
            return None
        return sources - uses

    def share(self, item: FlowItem, of_uses: bool) -> Optional[float]:
        """Returns an item's percentage of its column total."""
        total = self.total_uses if of_uses else self.total_sources
        return fmt.pos_margin(item.magnitude, total)


@dataclass
class ReinvestmentYear:
    """The reinvestment identity for one fiscal year."""

    period: str
    nopat: Optional[float] = None
    capex: Optional[float] = None
    depreciation: Optional[float] = None
    net_capex: Optional[float] = None
    delta_working_capital: Optional[float] = None
    reinvestment: Optional[float] = None
    reinvestment_rate: Optional[float] = None
    roic: Optional[float] = None
    implied_growth: Optional[float] = None
    revenue_growth: Optional[float] = None


@dataclass
class Reinvestment:
    """Whether reinvestment and returns account for the growth delivered.

    The identity is growth equals reinvestment rate times ROIC: a company
    can only compound at the rate it puts money back in, multiplied by what
    that money earns. Setting the implied rate beside the delivered rate is
    the check on whether reported growth was bought with capital or came
    from somewhere the accounts do not show.
    """

    years: List[ReinvestmentYear] = field(default_factory=list)
    window_from: str = ""
    window_to: str = ""
    total_reinvestment: Optional[float] = None
    total_nopat: Optional[float] = None
    aggregate_reinvestment_rate: Optional[float] = None
    average_roic: Optional[float] = None
    implied_growth: Optional[float] = None
    actual_revenue_cagr: Optional[float] = None
    actual_ebit_cagr: Optional[float] = None
    withheld_reason: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def growth_gap(self) -> Optional[float]:
        """Delivered revenue growth less the growth reinvestment implies."""
        if self.actual_revenue_cagr is None or self.implied_growth is None:
            return None
        return self.actual_revenue_cagr - self.implied_growth


@dataclass
class DupontYear:
    """The five-factor decomposition for one period, with its own check."""

    period: str
    tax_burden: Optional[float] = None
    interest_burden: Optional[float] = None
    operating_margin: Optional[float] = None
    asset_turnover: Optional[float] = None
    equity_multiplier: Optional[float] = None
    roe_direct: Optional[float] = None

    @property
    def factors(self) -> List[Optional[float]]:
        """The five factors in multiplication order."""
        return [self.tax_burden, self.interest_burden,
                self.operating_margin, self.asset_turnover,
                self.equity_multiplier]

    @property
    def roe_product(self) -> Optional[float]:
        """ROE rebuilt by multiplying the five factors, in percent."""
        values = self.factors
        if any(v is None for v in values):
            return None
        product = 1.0
        for value in values:
            product *= value
        # Operating margin is carried in percentage points, so the product
        # is already scaled to percent and needs no further factor of 100.
        return product

    @property
    def residual(self) -> Optional[float]:
        """Factor product less directly computed ROE, in percentage points.

        This is the decomposition auditing itself. A non-trivial residual
        means the five factors were not drawn from one consistent set of
        statements, and the row should not be trusted.
        """
        product, direct = self.roe_product, self.roe_direct
        if product is None or direct is None:
            return None
        return product - direct


@dataclass
class Dupont:
    """The DuPont decomposition across every reported period."""

    years: List[DupontYear] = field(default_factory=list)
    provider_period: str = ""
    provider_roe: Optional[float] = None
    provider_factors: Dict[str, Optional[float]] = field(default_factory=dict)
    reconciliation_delta: Optional[float] = None

    @property
    def first(self) -> Optional[DupontYear]:
        """Earliest period shown, the base for an indexed comparison."""
        return self.years[0] if self.years else None

    @property
    def latest(self) -> Optional[DupontYear]:
        """Most recent period."""
        return self.years[-1] if self.years else None


@dataclass
class Composites:
    """Every Tier 2 composite for one company."""

    piotroski: Piotroski
    piotroski_trend: List[PiotroskiPoint]
    altman: Altman
    altman_trend: List[AltmanPoint]
    sources_uses: SourcesAndUses
    reinvestment: Reinvestment
    dupont: Dupont
    is_financial: bool = False
    notes: List[str] = field(default_factory=list)


# --- helpers --------------------------------------------------------------


def _group_points(tests: Sequence[ScoreTest], low: int, high: int) -> Optional[int]:
    """Sums the points of the sub-tests numbered `low` through `high`."""
    awarded = [t.points for t in tests
               if low <= t.number <= high and t.points is not None]
    return sum(awarded) if awarded else None


def _sum_magnitudes(items: Sequence[FlowItem]) -> Optional[float]:
    """Totals a flow column, returning None if nothing is present."""
    values = [i.magnitude for i in items if i.magnitude is not None]
    return sum(values) if values else None


def _cagr(start: Optional[float], end: Optional[float],
          periods: int) -> Optional[float]:
    """Compound annual growth between two levels, in percentage points.

    Withheld unless both ends are strictly positive: a root taken through
    zero or from a negative base is not a growth rate.
    """
    if start is None or end is None or periods <= 0:
        return None
    if start <= 0 or end <= 0:
        return None
    return ((end / start) ** (1.0 / periods) - 1.0) * 100.0


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """Arithmetic mean of the present values, or None if there are none."""
    usable = [v for v in values if v is not None]
    return sum(usable) / len(usable) if usable else None


def is_financial(snap: CompanySnapshot) -> bool:
    """Reports whether the company is a lender or insurer.

    Two independent signals, either sufficient: the provider's sector label,
    and the absence of a consolidated total-debt line. The second catches a
    financial whose sector string is unhelpful, because a bank discloses
    deposits and borrowings separately and never a single debt figure.
    """
    sector = (snap.sector or "").lower()
    if any(marker in sector for marker in FINANCIAL_SECTOR_MARKERS):
        return True
    balances = [b for b in snap.balance if b.total_assets]
    if balances and all(b.debt is None for b in balances):
        return True
    return False


def _non_cash_working_capital(balance: BalancePeriod) -> Optional[float]:
    """Operating working capital, excluding cash and short-term debt.

    Both exclusions are deliberate. Cash and short-term borrowing are
    financing decisions rather than operating ones, and leaving them in
    makes the reinvestment figure move with treasury activity instead of
    with the business.

    The provider's own `cafCiwc` field is NOT used for this. It was tested
    against the balance sheet and does not reconcile: TCS reports roughly
    -17,000 crore in every one of the last four years against a
    balance-sheet movement of +168 to +4,781 crore, and a genuine
    working-capital delta oscillates rather than repeating one large
    negative. Whatever that field aggregates, it is not the year's change
    in working capital.
    """
    if balance.current_assets is None or balance.current_liabilities is None:
        return None
    assets = balance.current_assets - (balance.cash or 0.0)
    liabilities = balance.current_liabilities - (balance.short_term_debt or 0.0)
    return assets - liabilities


def _aligned(snap: CompanySnapshot) -> List[Tuple[IncomePeriod,
                                                  Optional[BalancePeriod],
                                                  Optional[CashflowPeriod]]]:
    """Pairs each fiscal year's income, balance and cash-flow rows.

    Matched on the period label rather than by position, because the three
    statements can carry different numbers of periods. TTM is excluded: the
    composites here need a full set of three statements and the provider
    publishes no trailing balance sheet or cash flow.
    """
    balance_by_period = {b.period: b for b in snap.balance}
    cash_by_period = {c.period: c for c in snap.cashflow}
    return [
        (income, balance_by_period.get(income.period), cash_by_period.get(income.period))
        for income in snap.years
    ]


# --- Piotroski F-Score ----------------------------------------------------


def _piotroski_at(
    rows: Sequence[Tuple[IncomePeriod, Optional[BalancePeriod], Optional[CashflowPeriod]]],
    position: int,
    financial: bool,
) -> Piotroski:
    """Scores the nine signals for the year at `position`.

    Three consecutive balance sheets are required, not two. Piotroski scales
    by OPENING assets, so the current year's return on assets needs last
    year's balance sheet and the PRIOR year's return on assets needs the one
    before that. Scoring the earliest years of a series is therefore not
    possible, which is correct rather than unfortunate: a change test with
    nothing to change from is not a test.

    Args:
        rows: Aligned income, balance and cash-flow rows, oldest first.
        position: Index of the year being scored.
        financial: Whether the company is a lender or insurer, in which case
            the leverage and liquidity signals are withheld as meaningless
            rather than computed as noise.

    Returns:
        A populated Piotroski record. Signals that cannot be evaluated carry
        a reason and contribute to neither the numerator nor the denominator
        of the score.
    """
    income, balance, cash = rows[position]
    prior_income, prior_balance, prior_cash = rows[position - 1]
    _, opening_prior_balance, _ = rows[position - 2]

    result = Piotroski(period=income.period, prior_period=prior_income.period)

    # Opening assets: last year's closing balance sheet for this year's
    # ratios, the one before it for last year's.
    opening = prior_balance.total_assets if prior_balance else None
    opening_prior = (opening_prior_balance.total_assets
                     if opening_prior_balance else None)

    roa = fmt.pos_margin(income.pat, opening)
    prior_roa = fmt.pos_margin(prior_income.pat, opening_prior)
    cfo = cash.cfo if cash else None
    turnover = fmt.pos_div(income.revenue, opening)
    prior_turnover = fmt.pos_div(prior_income.revenue, opening_prior)

    tests: List[ScoreTest] = []

    # --- profitability, signals 1 to 4 ---
    tests.append(ScoreTest(
        number=1, name="Return on assets positive",
        definition="PAT / opening total assets, versus nil",
        value=roa, comparator=0.0, unit="pct",
        passed=None if roa is None else roa > 0,
        unavailable_reason="" if roa is not None else
        "profit or opening total assets not reported",
    ))
    tests.append(ScoreTest(
        number=2, name="Operating cash flow positive",
        definition="Cash from operations, versus nil",
        value=cfo, comparator=0.0, unit="cr",
        passed=None if cfo is None else cfo > 0,
        unavailable_reason="" if cfo is not None else
        "cash flow statement not reported for this period",
    ))
    tests.append(ScoreTest(
        number=3, name="Return on assets improving",
        definition="This year's return on assets, versus last year's",
        value=roa, comparator=prior_roa, unit="pct",
        passed=None if (roa is None or prior_roa is None) else roa > prior_roa,
        unavailable_reason="" if (roa is not None and prior_roa is not None) else
        "two consecutive years of opening assets not available",
    ))
    tests.append(ScoreTest(
        number=4, name="Cash flow exceeds profit",
        definition="Cash from operations, versus PAT for the same year",
        value=cfo, comparator=income.pat, unit="cr",
        passed=None if (cfo is None or income.pat is None) else cfo > income.pat,
        unavailable_reason="" if (cfo is not None and income.pat is not None) else
        "cash flow or profit not reported",
    ))

    # --- leverage and liquidity, signals 5 to 7 ---
    if financial:
        reason = ("not meaningful for a financial: borrowings fund the asset "
                  "book rather than the operations, and customer deposits sit "
                  "in current liabilities")
        tests.append(ScoreTest(
            number=5, name="Leverage falling",
            definition="Long-term debt / average total assets, versus last year",
            unit="pct", passed=None, unavailable_reason=reason,
        ))
        tests.append(ScoreTest(
            number=6, name="Liquidity improving",
            definition="Current ratio, versus last year",
            unit="x", passed=None, unavailable_reason=reason,
        ))
        result.withheld.append(
            "Signals 5 and 6, on leverage and liquidity, are withheld: %s." % reason
        )
    else:
        average_assets = _average(
            balance.total_assets if balance else None, opening)
        prior_average_assets = _average(opening, opening_prior)
        leverage = fmt.pos_margin(
            balance.long_term_debt if balance else None, average_assets)
        prior_leverage = fmt.pos_margin(
            prior_balance.long_term_debt if prior_balance else None,
            prior_average_assets)
        tests.append(ScoreTest(
            number=5, name="Leverage falling",
            definition="Long-term debt / average total assets, versus last year",
            value=leverage, comparator=prior_leverage, unit="pct",
            passed=None if (leverage is None or prior_leverage is None)
            else leverage < prior_leverage,
            unavailable_reason="" if (leverage is not None and prior_leverage is not None)
            else "long-term debt not reported for both years",
        ))

        current_ratio = fmt.pos_div(
            balance.current_assets if balance else None,
            balance.current_liabilities if balance else None)
        prior_current_ratio = fmt.pos_div(
            prior_balance.current_assets if prior_balance else None,
            prior_balance.current_liabilities if prior_balance else None)
        tests.append(ScoreTest(
            number=6, name="Liquidity improving",
            definition="Current ratio, versus last year",
            value=current_ratio, comparator=prior_current_ratio, unit="x",
            passed=None if (current_ratio is None or prior_current_ratio is None)
            else current_ratio > prior_current_ratio,
            unavailable_reason="" if (current_ratio is not None
                                      and prior_current_ratio is not None)
            else "current assets or current liabilities not reported",
        ))

    shares = balance.shares_cr if balance else None
    prior_shares = prior_balance.shares_cr if prior_balance else None
    ceiling = (None if prior_shares is None
               else prior_shares * (1.0 + SHARE_ISSUE_TOLERANCE_PCT / 100.0))
    tests.append(ScoreTest(
        number=7, name="No equity raised",
        definition="Shares in issue, versus last year plus %.1f%% for option vesting"
                   % SHARE_ISSUE_TOLERANCE_PCT,
        value=shares, comparator=prior_shares, unit="cr",
        passed=None if (shares is None or ceiling is None) else shares <= ceiling,
        unavailable_reason="" if (shares is not None and ceiling is not None)
        else "share count not reported for both years",
    ))

    # --- operating efficiency, signals 8 and 9 ---
    # Signal 8 substitutes EBITDA margin for the paper's gross margin. See
    # the module docstring: this provider publishes no usable gross profit.
    margin_now = income.ebitda_margin
    margin_prior = prior_income.ebitda_margin
    tests.append(ScoreTest(
        number=8, name="Margin improving",
        definition="EBITDA margin, versus last year (paper uses gross margin)",
        value=margin_now, comparator=margin_prior, unit="pct",
        passed=None if (margin_now is None or margin_prior is None)
        else margin_now > margin_prior,
        unavailable_reason="" if (margin_now is not None and margin_prior is not None)
        else "EBITDA or revenue not reported for both years",
    ))
    tests.append(ScoreTest(
        number=9, name="Asset turnover improving",
        definition="Revenue / opening total assets, versus last year",
        value=turnover, comparator=prior_turnover, unit="x",
        passed=None if (turnover is None or prior_turnover is None)
        else turnover > prior_turnover,
        unavailable_reason="" if (turnover is not None and prior_turnover is not None)
        else "two consecutive years of opening assets not available",
    ))

    result.tests = tests
    result.substitutions.append(
        "Signal 8 uses EBITDA margin in place of the paper's gross margin. "
        "The provider reports no usable gross profit: the gross-profit field "
        "is null for every company tested, and the raw-material line it would "
        "have to be rebuilt from covers materials only, which is 64% of "
        "revenue for a refiner and under 1% for a services business."
    )
    result.substitutions.append(
        "Profitability and turnover are scaled by OPENING total assets, as "
        "the 2000 paper specifies. The returns section of this report uses "
        "period-end capital instead, to match the provider's own endpoints, "
        "so the return on assets shown here will not tie to it exactly."
    )
    return result


def _average(current: Optional[float], prior: Optional[float]) -> Optional[float]:
    """Two-point average, or None unless both points are present."""
    if current is None or prior is None:
        return None
    return (current + prior) / 2.0


def _piotroski_series(
    rows: Sequence[Tuple[IncomePeriod, Optional[BalancePeriod], Optional[CashflowPeriod]]],
    financial: bool,
) -> List[PiotroskiPoint]:
    """Scores every year that has the three balance sheets it needs."""
    out: List[PiotroskiPoint] = []
    for position in range(2, len(rows)):
        scored = _piotroski_at(rows, position, financial)
        out.append(PiotroskiPoint(
            period=scored.period,
            score=scored.score,
            computable=scored.computable,
        ))
    return out


# --- Altman Z-Score -------------------------------------------------------


def _altman_components(
    income: IncomePeriod,
    balance: BalancePeriod,
    equity_value: Optional[float],
    equity_label: str,
    weights: Dict[str, float],
) -> List[AltmanComponent]:
    """Builds the five weighted terms of a Z-Score.

    Total assets and total liabilities are the denominators throughout, and
    both are taken through the positive-only guard: a negative or absent
    asset base does not produce a low score, it produces no score.
    """
    assets = balance.total_assets
    liabilities = balance.total_liabilities
    return [
        AltmanComponent(
            key="working_capital", name="Working capital / assets",
            definition="(Current assets less current liabilities) / total assets",
            ratio=fmt.pos_div(balance.working_capital, assets),
            weight=weights["working_capital"],
        ),
        AltmanComponent(
            key="retained_earnings", name="Retained earnings / assets",
            definition="Accumulated retained earnings / total assets",
            ratio=fmt.pos_div(balance.retained_earnings, assets),
            weight=weights["retained_earnings"],
        ),
        AltmanComponent(
            key="ebit", name="EBIT / assets",
            definition="Operating profit / total assets",
            ratio=fmt.pos_div(income.ebit, assets),
            weight=weights["ebit"],
        ),
        AltmanComponent(
            key="equity_value", name="%s / liabilities" % equity_label,
            definition="%s / total liabilities" % equity_label,
            ratio=fmt.pos_div(equity_value, liabilities),
            weight=weights["equity_value"],
        ),
        AltmanComponent(
            key="sales", name="Revenue / assets",
            definition="Revenue / total assets",
            ratio=fmt.pos_div(income.revenue, assets),
            weight=weights["sales"],
        ),
    ]


def _altman(snap: CompanySnapshot, financial: bool) -> Altman:
    """Builds the listed-company Z-Score for the latest reported year.

    Market capitalisation supplies the equity term, which is what makes this
    the 1968 listed model rather than the 1983 book-value revision. The
    figure is a current one against a balance sheet dated at the fiscal year
    end; that timing mismatch is inherent to the model whenever it is run on
    live prices, and the equity basis is labelled so a reader can see it.
    """
    if financial:
        return Altman(withheld_reason=(
            "The Z-Score is withheld for financials. Altman's model was fitted "
            "on manufacturers and two of its five terms do not describe a "
            "lender: working capital is not a meaningful concept where customer "
            "deposits sit in current liabilities, and revenue over assets "
            "measures balance-sheet turnover rather than operating efficiency. "
            "A number can be produced; it would not mean anything."
        ))

    balance = snap.latest_balance
    if balance is None:
        return Altman(withheld_reason="No balance sheet reported.")
    income = next((y for y in reversed(snap.years) if y.period == balance.period), None)
    if income is None:
        return Altman(withheld_reason=(
            "No income statement matches the latest balance-sheet period."))

    equity_value = snap.market_cap_cr or snap.implied_market_cap_cr
    basis = ("Market capitalisation" if snap.market_cap_cr
             else "Implied market capitalisation")
    return Altman(
        period=balance.period,
        model="Altman Z (1968), listed-company model",
        equity_basis=basis,
        components=_altman_components(
            income, balance, equity_value, basis, ALTMAN_PUBLIC_WEIGHTS),
        safe_above=ALTMAN_PUBLIC_SAFE,
        distress_below=ALTMAN_PUBLIC_DISTRESS,
    )


def _altman_series(snap: CompanySnapshot, financial: bool) -> List[AltmanPoint]:
    """Builds the Z-prime series across every reported year.

    The historical series has to use Z-prime, the 1983 revision, because no
    share price is available for a past year and the equity term must be
    taken at book. Its coefficients and cut-offs differ from the listed
    model's, so the level of this series is NOT comparable with the
    latest-year Z above it, only its direction is. The report states that
    where the two appear together.
    """
    if financial:
        return []
    balance_by_period = {b.period: b for b in snap.balance}
    out: List[AltmanPoint] = []
    for income in snap.years:
        balance = balance_by_period.get(income.period)
        if balance is None:
            continue
        score = Altman(
            period=income.period,
            components=_altman_components(
                income, balance, balance.equity, "Book equity",
                ALTMAN_PRIVATE_WEIGHTS),
            safe_above=ALTMAN_PRIVATE_SAFE,
            distress_below=ALTMAN_PRIVATE_DISTRESS,
        )
        out.append(AltmanPoint(
            period=income.period, score=score.score, zone=score.zone))
    return out


# --- capital allocation: sources and uses ---------------------------------


def _sources_and_uses(snap: CompanySnapshot, window: int) -> SourcesAndUses:
    """Aggregates the cash-flow statement into a sources-and-uses statement.

    The construction rests on the statement's own articulation: operating
    plus investing plus financing equals the change in cash. Rearranged,
    every inflow is a source and every outflow a use, and the change in the
    cash balance closes the difference. Because that is an identity rather
    than an estimate, the two columns balance exactly, and `selfcheck`
    asserts it for every report.

    Two lines are residuals rather than reported figures, and are labelled as
    such on the page: investing beyond capital expenditure, and financing
    beyond dividends. Neither can be decomposed further, because the provider
    publishes no debt-raised, debt-repaid or buyback line - only the net
    financing and net investing totals.

    Args:
        snap: Populated company snapshot.
        window: Maximum number of fiscal years to accumulate.

    Returns:
        A SourcesAndUses record. Years missing any of the five inputs are
        excluded from the accumulation rather than treated as nil, and the
        window actually used is recorded on the result.
    """
    usable = [
        c for c in snap.cashflow
        if None not in (c.cfo, c.capex, c.fcf, c.dividends_paid,
                        c.cash_from_investing, c.cash_from_financing)
    ][-window:]
    if not usable:
        return SourcesAndUses(notes=[
            "The provider reports no cash flow statement for this company, so "
            "capital allocation cannot be traced."])

    total_cfo = sum(c.cfo for c in usable)
    total_capex = sum(c.capex for c in usable)
    total_dividends = sum(c.dividends_paid for c in usable)
    total_investing = sum(c.cash_from_investing for c in usable)
    total_financing = sum(c.cash_from_financing for c in usable)

    # Capex and dividends are reported as positive outflow magnitudes, while
    # the investing and financing totals are signed. Adding the magnitude
    # back to its signed total leaves everything else in that section.
    other_investing = total_investing + total_capex
    other_financing = total_financing + total_dividends
    net_cash_change = total_cfo + total_investing + total_financing

    result = SourcesAndUses(
        from_period=usable[0].period,
        to_period=usable[-1].period,
        years=len(usable),
    )

    def place(amount: float, source_label: str, use_label: str,
              definition: str) -> None:
        """Files one net flow into whichever column its sign belongs to."""
        if amount == 0:
            return
        item = FlowItem(label=source_label if amount > 0 else use_label,
                        amount=amount, definition=definition)
        (result.sources if amount > 0 else result.uses).append(item)

    place(total_cfo, "Operating cash flow", "Operating cash outflow",
          "Cash generated by operations, before capital expenditure")
    place(-total_capex, "Capital expenditure recovered", "Capital expenditure",
          "Purchases of property, plant, equipment and intangibles")
    place(other_investing,
          "Net disposals and investment maturities",
          "Net acquisitions and investments",
          "Investing cash flow other than capital expenditure; a residual")
    place(-total_dividends, "Dividends received", "Dividends paid",
          "Cash dividends paid to shareholders")
    # Deliberately neutral about what the residual contains. Calling the
    # outflow "net debt repaid" would have been wrong for Wipro, where the
    # line is 44,374 crore of outflow while total debt ROSE by 10,567: the
    # money went to buybacks and lease payments, not to retiring debt. The
    # statement does not separate them, so the label must not pretend it does.
    place(other_financing,
          "Net financing raised, other than dividends",
          "Net financing returned, other than dividends",
          "Financing cash flow other than dividends; a residual covering debt "
          "drawn and repaid, buybacks and lease payments together")
    place(-net_cash_change,
          "Cash and equivalents drawn down",
          "Cash and equivalents built up",
          "The closing balance the other lines leave behind")

    balances = [b for b in snap.balance
                if b.period in {c.period for c in usable}]
    if len(balances) >= 2:
        if balances[0].debt is not None and balances[-1].debt is not None:
            result.debt_change = balances[-1].debt - balances[0].debt
        if balances[0].shares_cr and balances[-1].shares_cr:
            result.share_change_pct = fmt.pos_margin(
                balances[-1].shares_cr - balances[0].shares_cr,
                balances[0].shares_cr)

    if len(usable) < min(window, len(snap.cashflow)):
        result.notes.append(
            "Accumulated over the %d years with a complete cash flow "
            "statement; earlier years are missing one or more lines."
            % len(usable))
    return result


# --- capital allocation: the reinvestment identity ------------------------


def _reinvestment(
    snap: CompanySnapshot,
    derived: DerivedAnalytics,
    financial: bool,
    window: int,
) -> Reinvestment:
    """Tests growth against the reinvestment that should have produced it.

    Growth equals the reinvestment rate multiplied by the return on that
    reinvestment. Both sides of the identity are computed here from the
    statements and set against the revenue and EBIT growth actually
    delivered, which is the check on whether reported growth was paid for.

    Reinvestment is capital expenditure net of depreciation plus the increase
    in non-cash working capital: what the business put in beyond replacing
    what wore out. Depreciation is treated as maintenance capital, which is
    the standard simplification and is stated on the page.
    """
    if financial:
        return Reinvestment(withheld_reason=(
            "The reinvestment identity is withheld for financials. It measures "
            "growth bought with fixed assets and working capital, and a lender "
            "grows by deploying its balance sheet instead: capital expenditure "
            "is incidental to the business and working capital is not a "
            "meaningful concept where deposits are a current liability."
        ))

    metrics_by_period: Dict[str, AnnualMetrics] = {
        m.period: m for m in derived.annual}
    balance_by_period = {b.period: b for b in snap.balance}
    cash_by_period = {c.period: c for c in snap.cashflow}

    years: List[ReinvestmentYear] = []
    for position, income in enumerate(snap.years):
        if position == 0:
            continue
        balance = balance_by_period.get(income.period)
        prior_balance = balance_by_period.get(snap.years[position - 1].period)
        cash = cash_by_period.get(income.period)
        metrics = metrics_by_period.get(income.period)
        if balance is None or prior_balance is None or cash is None or metrics is None:
            continue

        working_capital = _non_cash_working_capital(balance)
        prior_working_capital = _non_cash_working_capital(prior_balance)
        delta_working_capital = (
            None if (working_capital is None or prior_working_capital is None)
            else working_capital - prior_working_capital)

        net_capex = (None if (cash.capex is None or income.depreciation is None)
                     else cash.capex - income.depreciation)
        reinvestment = (None if (net_capex is None or delta_working_capital is None)
                        else net_capex + delta_working_capital)
        rate = fmt.pos_margin(reinvestment, metrics.nopat)
        implied = (None if (rate is None or metrics.roic is None)
                   else rate / 100.0 * metrics.roic)

        years.append(ReinvestmentYear(
            period=income.period,
            nopat=metrics.nopat,
            capex=cash.capex,
            depreciation=income.depreciation,
            net_capex=net_capex,
            delta_working_capital=delta_working_capital,
            reinvestment=reinvestment,
            reinvestment_rate=rate,
            roic=metrics.roic,
            implied_growth=implied,
            revenue_growth=fmt.growth(
                income.revenue, snap.years[position - 1].revenue),
        ))

    years = years[-window:]
    if not years:
        return Reinvestment(withheld_reason=(
            "Two consecutive years of balance sheet and cash flow are needed "
            "to measure reinvestment, and the provider reports fewer."))

    result = Reinvestment(
        years=years,
        window_from=years[0].period,
        window_to=years[-1].period,
        average_roic=_mean([y.roic for y in years]),
    )

    # The window rate is aggregated, not averaged. A mean of annual rates
    # lets one restructuring year dominate the answer: Reliance's FY2021
    # working-capital swing alone produces a 234% annual rate, which drags a
    # seven-year mean far from anything the company actually did. Summing
    # both sides first weights each year by its own size, which is what the
    # identity is asking about.
    reinvestments = [y.reinvestment for y in years if y.reinvestment is not None]
    nopats = [y.nopat for y in years if y.nopat is not None]
    if reinvestments and len(reinvestments) == len(nopats):
        result.total_reinvestment = sum(reinvestments)
        result.total_nopat = sum(nopats)
        result.aggregate_reinvestment_rate = fmt.pos_margin(
            result.total_reinvestment, result.total_nopat)
    if result.aggregate_reinvestment_rate is not None and result.average_roic is not None:
        result.implied_growth = (
            result.aggregate_reinvestment_rate / 100.0 * result.average_roic)

    # Growth is measured across the same window the identity was averaged
    # over, so the two sides describe one period rather than two.
    window_periods = {y.period for y in years}
    income_window = [y for y in snap.years if y.period in window_periods]
    if len(income_window) >= 2:
        spans = len(income_window) - 1
        result.actual_revenue_cagr = _cagr(
            income_window[0].revenue, income_window[-1].revenue, spans)
        result.actual_ebit_cagr = _cagr(
            income_window[0].ebit, income_window[-1].ebit, spans)

    negative_nopat = [y.period for y in years
                      if y.reinvestment is not None and y.reinvestment_rate is None]
    if negative_nopat:
        result.notes.append(
            "The reinvestment rate is withheld for %s: net operating profit "
            "after tax was nil or negative, and a rate against that base "
            "carries no meaning." % ", ".join(fmt.period_label(p)
                                              for p in negative_nopat))

    # The identity only sees investment that lands on the balance sheet. A
    # business that grows by hiring, or by spending on research and brand,
    # expenses all of it, so reinvestment reads at or below zero while
    # revenue compounds. Saying so is the difference between an exhibit that
    # informs and one that looks broken.
    net_capex_values = [y.net_capex for y in years if y.net_capex is not None]
    mostly_negative = (
        len(net_capex_values) >= 3
        and sum(1 for v in net_capex_values if v < 0) > len(net_capex_values) / 2.0
    )
    if mostly_negative:
        result.notes.append(
            "Capital expenditure ran below depreciation in most years of the "
            "window, so measured reinvestment is negative and the implied "
            "growth rate falls below the growth delivered. That gap is a "
            "property of the measure rather than a contradiction: this "
            "identity counts only investment that is capitalised onto the "
            "balance sheet, and a business that grows through headcount, "
            "research or brand expenses that spending as it goes. Read the "
            "gap as the share of growth this framework cannot attribute."
        )
    elif (result.growth_gap is not None and abs(result.growth_gap) > 5.0):
        result.notes.append(
            "Delivered revenue growth differs from the growth reinvestment "
            "implies by %s. A positive gap points to growth from sources this "
            "identity does not capture, such as pricing, acquisitions "
            "accounted for outside capital expenditure, or expensed "
            "investment; a negative gap means capital went in without "
            "commensurate growth coming out."
            % fmt.signed_pct(result.growth_gap)
        )
    return result


# --- DuPont across all years ----------------------------------------------


def _dupont(snap: CompanySnapshot) -> Dupont:
    """Decomposes return on equity into five factors, for every period.

    The provider publishes this decomposition for one trailing period only.
    A single column says where returns are; the series says which factor
    moved them, which is the whole reason the decomposition exists.

    Each row also rebuilds ROE by multiplying its own five factors and sets
    that against ROE computed directly from profit and equity. The residual
    is the row auditing itself: anything beyond rounding means the factors
    were not drawn from one consistent set of statements.
    """
    balance_by_period = {b.period: b for b in snap.balance}
    periods = list(snap.years) + ([snap.ttm] if snap.ttm else [])

    rows: List[DupontYear] = []
    for income in periods:
        balance = balance_by_period.get(income.period)
        # The trailing row has no balance sheet of its own; the latest
        # reported one is its correct pairing, as elsewhere in the report.
        if balance is None and income is snap.ttm and snap.balance:
            balance = snap.balance[-1]
        if balance is None:
            continue
        rows.append(DupontYear(
            period=income.period,
            tax_burden=fmt.pos_div(income.pat, income.pbt),
            interest_burden=fmt.pos_div(income.pbt, income.ebit),
            operating_margin=fmt.pos_margin(income.ebit, income.revenue),
            asset_turnover=fmt.pos_div(income.revenue, balance.total_assets),
            equity_multiplier=fmt.pos_div(balance.total_assets, balance.equity),
            roe_direct=fmt.pos_margin(income.pat, balance.equity),
        ))

    result = Dupont(years=rows)

    provider = snap.dupont or {}
    if provider:
        result.provider_period = str(provider.get("period") or "")
        result.provider_roe = provider.get("return_on_equity_roe_pct")
        result.provider_factors = {
            "tax_burden": provider.get("tax_burden_ratio"),
            "interest_burden": provider.get("interest_burden_ratio"),
            "operating_margin": provider.get("operating_margin_pct"),
            "asset_turnover": provider.get("asset_turnover_x"),
            "equity_multiplier": provider.get("equity_multiplier_x"),
        }
        match = next((r for r in rows
                      if r.period.strip().upper()
                      == result.provider_period.strip().upper()), None)
        if match is not None and match.roe_product is not None \
                and result.provider_roe is not None:
            result.reconciliation_delta = match.roe_product - result.provider_roe
    return result


# --- entry point ----------------------------------------------------------


def compute(
    snap: CompanySnapshot,
    derived: DerivedAnalytics,
    window: int = tokens.DISPLAY_YEARS,
) -> Composites:
    """Computes every Tier 2 composite for one company.

    Args:
        snap: Normalised company snapshot.
        derived: Tier 1 analytics, which supply NOPAT and ROIC to the
            reinvestment identity rather than having them recomputed here.
        window: Fiscal years to accumulate or display, matched to the
            report's chart and table window so every exhibit covers the
            same periods.

    Returns:
        A Composites record. Frameworks that do not apply to the company
        carry a withholding reason instead of a number, and `notes` collects
        anything the report should state about what was withheld and why.
    """
    financial = is_financial(snap)
    rows = _aligned(snap)

    if len(rows) >= 3:
        piotroski = _piotroski_at(rows, len(rows) - 1, financial)
        trend = _piotroski_series(rows, financial)
    else:
        piotroski = Piotroski(withheld=[
            "The F-Score needs three consecutive years of statements, because "
            "its change tests are scaled by opening assets. The provider "
            "reports %d." % len(rows)])
        trend = []

    composites = Composites(
        piotroski=piotroski,
        piotroski_trend=trend,
        altman=_altman(snap, financial),
        altman_trend=_altman_series(snap, financial),
        sources_uses=_sources_and_uses(snap, window),
        reinvestment=_reinvestment(snap, derived, financial, window),
        dupont=_dupont(snap),
        is_financial=financial,
    )

    if financial:
        composites.notes.append(
            "This company is classified as a financial, so the Z-Score and the "
            "reinvestment identity are withheld and two of the nine F-Score "
            "signals are not evaluated. Each exhibit states its own reason."
        )
    if piotroski.tests and not piotroski.comparable:
        composites.notes.append(
            "The F-Score is reported out of %d evaluated signals rather than "
            "nine and is therefore not comparable with a standard F-Score."
            % piotroski.computable
        )
    return composites
