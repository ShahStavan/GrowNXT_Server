"""Tier 2 composites: scored frameworks and identity-based decompositions.

Two published scoring frameworks, two cash-allocation identities, and DuPont
across every reported year. One rule shapes the module: **a composite is
never reported as a bare number** -- each carries its own components as data,
and the score is a derived property of them. `.claude/specs/reporting-quantitative-engine.md` section 5.
"""

import logging
import operator
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

from reporting import fmt, tokens
from reporting.analytics import AnnualMetrics, DerivedAnalytics
from reporting.snapshot import (
    BalancePeriod,
    CashflowPeriod,
    CompanySnapshot,
    IncomePeriod,
)

logger = logging.getLogger(__name__)

# Sector strings that mark a lender or insurer. Matched case-insensitively
# as substrings against the provider's sector label, which arrives as free
# text such as 'Private Banks' or 'Insurance'.
FINANCIAL_SECTOR_MARKERS: tuple[str, ...] = (
    "bank",
    "financ",
    "nbfc",
    "insur",
    "capital market",
    "asset management",
    "broker",
    "lending",
    "housing finance",
)

# Piotroski's share-issuance signal asks whether the company raised equity.
# Only the share COUNT is available here, and a count drifts upward every
# year on employee-option vesting at companies that have raised nothing.
# Growth below this threshold is read as that drift rather than a raise.
SHARE_ISSUE_TOLERANCE_PCT: float = 0.5

# Altman (1968) coefficients and cut-offs, the listed-company model.
ALTMAN_PUBLIC_WEIGHTS: dict[str, float] = {
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
ALTMAN_PRIVATE_WEIGHTS: dict[str, float] = {
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
    value: float | None = None
    comparator: float | None = None
    unit: str = "pct"
    passed: bool | None = None
    unavailable_reason: str = ""

    @property
    def points(self) -> int | None:
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
    tests: list[ScoreTest] = field(default_factory=list)
    substitutions: list[str] = field(default_factory=list)
    withheld: list[str] = field(default_factory=list)

    @property
    def score(self) -> int | None:
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
    def profitability_points(self) -> int | None:
        """Points from signals 1-4, the profitability group."""
        return _group_points(self.tests, 1, 4)

    @property
    def leverage_points(self) -> int | None:
        """Points from signals 5-7, the leverage and liquidity group."""
        return _group_points(self.tests, 5, 7)

    @property
    def efficiency_points(self) -> int | None:
        """Points from signals 8-9, the operating-efficiency group."""
        return _group_points(self.tests, 8, 9)


@dataclass
class PiotroskiPoint:
    """One period of the F-Score series."""

    period: str
    score: int | None = None
    computable: int = 0


@dataclass
class AltmanComponent:
    """One weighted term of a Z-Score."""

    key: str
    name: str
    definition: str
    ratio: float | None = None
    weight: float = 0.0

    @property
    def contribution(self) -> float | None:
        """The term's contribution to the score."""
        return None if self.ratio is None else self.ratio * self.weight


@dataclass
class Altman:
    """A Z-Score with the five terms that produced it."""

    period: str = ""
    model: str = ""
    equity_basis: str = ""
    components: list[AltmanComponent] = field(default_factory=list)
    safe_above: float = ALTMAN_PUBLIC_SAFE
    distress_below: float = ALTMAN_PUBLIC_DISTRESS
    withheld_reason: str = ""

    @property
    def score(self) -> float | None:
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
    def dominant(self) -> AltmanComponent | None:
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
    def concentration_pct(self) -> float | None:
        """Share of the score contributed by its largest single term."""
        component = self.dominant
        if component is None:
            return None
        return fmt.pos_margin(component.contribution, self.score)


@dataclass
class AltmanPoint:
    """One period of the Z-prime series."""

    period: str
    score: float | None = None
    zone: str = ""


@dataclass
class FlowItem:
    """One line of a sources-and-uses statement."""

    label: str
    amount: float | None = None
    definition: str = ""

    @property
    def magnitude(self) -> float | None:
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
    sources: list[FlowItem] = field(default_factory=list)
    uses: list[FlowItem] = field(default_factory=list)
    debt_change: float | None = None
    share_change_pct: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def total_sources(self) -> float | None:
        """Sum of the sources column."""
        return _sum_magnitudes(self.sources)

    @property
    def total_uses(self) -> float | None:
        """Sum of the uses column."""
        return _sum_magnitudes(self.uses)

    @property
    def residual(self) -> float | None:
        """Sources less uses. Must be zero; `selfcheck` asserts that."""
        sources, uses = self.total_sources, self.total_uses
        if sources is None or uses is None:
            return None
        return sources - uses

    def share(self, item: FlowItem, of_uses: bool) -> float | None:
        """Returns an item's percentage of its column total."""
        total = self.total_uses if of_uses else self.total_sources
        return fmt.pos_margin(item.magnitude, total)


@dataclass
class ReinvestmentYear:
    """The reinvestment identity for one fiscal year."""

    period: str
    nopat: float | None = None
    capex: float | None = None
    depreciation: float | None = None
    net_capex: float | None = None
    delta_working_capital: float | None = None
    reinvestment: float | None = None
    reinvestment_rate: float | None = None
    roic: float | None = None
    implied_growth: float | None = None
    revenue_growth: float | None = None


@dataclass
class Reinvestment:
    """Whether reinvestment and returns account for the growth delivered.

    The identity is growth equals reinvestment rate times ROIC: a company
    can only compound at the rate it puts money back in, multiplied by what
    that money earns. Setting the implied rate beside the delivered rate is
    the check on whether reported growth was bought with capital or came
    from somewhere the accounts do not show.
    """

    years: list[ReinvestmentYear] = field(default_factory=list)
    window_from: str = ""
    window_to: str = ""
    total_reinvestment: float | None = None
    total_nopat: float | None = None
    aggregate_reinvestment_rate: float | None = None
    average_roic: float | None = None
    implied_growth: float | None = None
    actual_revenue_cagr: float | None = None
    actual_ebit_cagr: float | None = None
    withheld_reason: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def growth_gap(self) -> float | None:
        """Delivered revenue growth less the growth reinvestment implies."""
        if self.actual_revenue_cagr is None or self.implied_growth is None:
            return None
        return self.actual_revenue_cagr - self.implied_growth


@dataclass
class DupontYear:
    """The five-factor decomposition for one period, with its own check."""

    period: str
    tax_burden: float | None = None
    interest_burden: float | None = None
    operating_margin: float | None = None
    asset_turnover: float | None = None
    equity_multiplier: float | None = None
    roe_direct: float | None = None

    @property
    def factors(self) -> list[float | None]:
        """The five factors in multiplication order."""
        return [
            self.tax_burden,
            self.interest_burden,
            self.operating_margin,
            self.asset_turnover,
            self.equity_multiplier,
        ]

    @property
    def roe_product(self) -> float | None:
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
    def residual(self) -> float | None:
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

    years: list[DupontYear] = field(default_factory=list)
    provider_period: str = ""
    provider_roe: float | None = None
    provider_factors: dict[str, float | None] = field(default_factory=dict)
    reconciliation_delta: float | None = None

    @property
    def first(self) -> DupontYear | None:
        """Earliest period shown, the base for an indexed comparison."""
        return self.years[0] if self.years else None

    @property
    def latest(self) -> DupontYear | None:
        """Most recent period."""
        return self.years[-1] if self.years else None


@dataclass
class Composites:
    """Every Tier 2 composite for one company."""

    piotroski: Piotroski
    piotroski_trend: list[PiotroskiPoint]
    altman: Altman
    altman_trend: list[AltmanPoint]
    sources_uses: SourcesAndUses
    reinvestment: Reinvestment
    dupont: Dupont
    is_financial: bool = False
    notes: list[str] = field(default_factory=list)


# --- helpers --------------------------------------------------------------


def _group_points(tests: Sequence[ScoreTest], low: int, high: int) -> int | None:
    """Sums the points of the sub-tests numbered `low` through `high`."""
    awarded = [
        t.points for t in tests if low <= t.number <= high and t.points is not None
    ]
    return sum(awarded) if awarded else None


def _sum_magnitudes(items: Sequence[FlowItem]) -> float | None:
    """Totals a flow column, returning None if nothing is present."""
    values = [i.magnitude for i in items if i.magnitude is not None]
    return sum(values) if values else None


def _cagr(start: float | None, end: float | None, periods: int) -> float | None:
    """Compound annual growth between two levels, in percentage points.

    Withheld unless both ends are strictly positive: a root taken through
    zero or from a negative base is not a growth rate.
    """
    if start is None or end is None or periods <= 0:
        return None
    if start <= 0 or end <= 0:
        return None
    return ((end / start) ** (1.0 / periods) - 1.0) * 100.0


def _mean(values: Sequence[float | None]) -> float | None:
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
    return bool(balances) and all(b.debt is None for b in balances)


def _non_cash_working_capital(balance: BalancePeriod) -> float | None:
    """Operating working capital, excluding cash and short-term debt.

    Both exclusions are deliberate: cash and short-term borrowing are
    financing decisions, and leaving them in makes reinvestment move with
    treasury activity rather than the business.

    The provider's `cafCiwc` is NOT used -- it does not reconcile. TCS reports
    about -17,000 crore in each of the last four years against a balance-sheet
    movement of +168 to +4,781 crore, and a real working-capital delta
    oscillates rather than repeating one large negative.
    """
    if balance.current_assets is None or balance.current_liabilities is None:
        return None
    assets = balance.current_assets - (balance.cash or 0.0)
    liabilities = balance.current_liabilities - (balance.short_term_debt or 0.0)
    return assets - liabilities


def _aligned(
    snap: CompanySnapshot,
) -> list[tuple[IncomePeriod, BalancePeriod | None, CashflowPeriod | None]]:
    """Pairs each fiscal year's income, balance and cash-flow rows.

    Matched on the period label rather than by position, because the three
    statements can carry different numbers of periods. TTM is excluded: the
    composites here need a full set of three statements and the provider
    publishes no trailing balance sheet or cash flow.
    """
    balance_by_period = {b.period: b for b in snap.balance}
    cash_by_period = {c.period: c for c in snap.cashflow}
    return [
        (
            income,
            balance_by_period.get(income.period),
            cash_by_period.get(income.period),
        )
        for income in snap.years
    ]


# --- Piotroski F-Score ----------------------------------------------------

# Comparison a signal has to satisfy to earn its point.
UP = operator.gt
DOWN = operator.lt
AT_MOST = operator.le


def _signal(
    number: int,
    name: str,
    definition: str,
    unit: str,
    value: float | None,
    comparator: float | None,
    better,
    reason: str,
    limit: float | None = None,
) -> ScoreTest:
    """Builds one F-Score signal, deriving availability and verdict once.

    Every signal has the same shape: a figure, something to beat, and a
    direction. Repeating the two None-checks per signal is how a signal ends
    up scored against a missing base -- which reads on the page as a point
    earned rather than one that could not be evaluated.

    Args:
        number: Signal number, 1 to 9.
        name: Short label for the exhibit.
        definition: What was measured, as printed.
        unit: Unit of `value`, for formatting.
        value: This period's figure.
        comparator: The figure shown as the thing to beat.
        better: Comparison the signal must satisfy: UP, DOWN or AT_MOST.
        reason: Why the signal could not be evaluated, used when it cannot.
        limit: Threshold actually compared against, when it differs from the
            comparator shown -- signal 7 displays last year's share count but
            allows a tolerance above it.

    Returns:
        The signal, scored or explicitly unavailable.

    """
    known = value is not None and comparator is not None
    return ScoreTest(
        number=number,
        name=name,
        definition=definition,
        value=value,
        comparator=comparator,
        unit=unit,
        passed=better(value, comparator if limit is None else limit) if known else None,
        unavailable_reason="" if known else reason,
    )


def _piotroski_at(
    rows: Sequence[tuple[IncomePeriod, BalancePeriod | None, CashflowPeriod | None]],
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
    opening_prior = (
        opening_prior_balance.total_assets if opening_prior_balance else None
    )

    roa = fmt.pos_margin(income.pat, opening)
    prior_roa = fmt.pos_margin(prior_income.pat, opening_prior)
    cfo = cash.cfo if cash else None
    turnover = fmt.pos_div(income.revenue, opening)
    prior_turnover = fmt.pos_div(prior_income.revenue, opening_prior)

    tests: list[ScoreTest] = []

    # --- profitability, signals 1 to 4 ---
    tests.append(
        _signal(
            1,
            "Return on assets positive",
            "PAT / opening total assets, versus nil",
            "pct",
            roa,
            0.0,
            UP,
            "profit or opening total assets not reported",
        )
    )
    tests.append(
        _signal(
            2,
            "Operating cash flow positive",
            "Cash from operations, versus nil",
            "cr",
            cfo,
            0.0,
            UP,
            "cash flow statement not reported for this period",
        )
    )
    tests.append(
        _signal(
            3,
            "Return on assets improving",
            "This year's return on assets, versus last year's",
            "pct",
            roa,
            prior_roa,
            UP,
            "two consecutive years of opening assets not available",
        )
    )
    tests.append(
        _signal(
            4,
            "Cash flow exceeds profit",
            "Cash from operations, versus PAT for the same year",
            "cr",
            cfo,
            income.pat,
            UP,
            "cash flow or profit not reported",
        )
    )

    # --- leverage and liquidity, signals 5 to 7 ---
    if financial:
        reason = (
            "not meaningful for a financial: borrowings fund the asset "
            "book rather than the operations, and customer deposits sit "
            "in current liabilities"
        )
        tests.append(
            _signal(
                5,
                "Leverage falling",
                "Long-term debt / average total assets, versus last year",
                "pct",
                None,
                None,
                DOWN,
                reason,
            )
        )
        tests.append(
            _signal(
                6,
                "Liquidity improving",
                "Current ratio, versus last year",
                "x",
                None,
                None,
                UP,
                reason,
            )
        )
        result.withheld.append(
            f"Signals 5 and 6, on leverage and liquidity, are withheld: {reason}."
        )
    else:
        average_assets = _average(balance.total_assets if balance else None, opening)
        prior_average_assets = _average(opening, opening_prior)
        leverage = fmt.pos_margin(
            balance.long_term_debt if balance else None, average_assets
        )
        prior_leverage = fmt.pos_margin(
            prior_balance.long_term_debt if prior_balance else None,
            prior_average_assets,
        )
        tests.append(
            _signal(
                5,
                "Leverage falling",
                "Long-term debt / average total assets, versus last year",
                "pct",
                leverage,
                prior_leverage,
                DOWN,
                "long-term debt not reported for both years",
            )
        )

        current_ratio = fmt.pos_div(
            balance.current_assets if balance else None,
            balance.current_liabilities if balance else None,
        )
        prior_current_ratio = fmt.pos_div(
            prior_balance.current_assets if prior_balance else None,
            prior_balance.current_liabilities if prior_balance else None,
        )
        tests.append(
            _signal(
                6,
                "Liquidity improving",
                "Current ratio, versus last year",
                "x",
                current_ratio,
                prior_current_ratio,
                UP,
                "current assets or current liabilities not reported",
            )
        )

    shares = balance.shares_cr if balance else None
    prior_shares = prior_balance.shares_cr if prior_balance else None
    ceiling = (
        None
        if prior_shares is None
        else prior_shares * (1.0 + SHARE_ISSUE_TOLERANCE_PCT / 100.0)
    )
    tests.append(
        _signal(
            7,
            "No equity raised",
            f"Shares in issue, versus last year plus {SHARE_ISSUE_TOLERANCE_PCT:.1f}% "
            "for option vesting",
            "cr",
            shares,
            prior_shares,
            AT_MOST,
            "share count not reported for both years",
            limit=ceiling,
        )
    )

    # --- operating efficiency, signals 8 and 9 ---
    # Signal 8 substitutes EBITDA margin for the paper's gross margin. See
    # the module docstring: this provider publishes no usable gross profit.
    margin_now = income.ebitda_margin
    margin_prior = prior_income.ebitda_margin
    tests.append(
        _signal(
            8,
            "Margin improving",
            "EBITDA margin, versus last year (paper uses gross margin)",
            "pct",
            margin_now,
            margin_prior,
            UP,
            "EBITDA or revenue not reported for both years",
        )
    )
    tests.append(
        _signal(
            9,
            "Asset turnover improving",
            "Revenue / opening total assets, versus last year",
            "x",
            turnover,
            prior_turnover,
            UP,
            "two consecutive years of opening assets not available",
        )
    )

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


def _average(current: float | None, prior: float | None) -> float | None:
    """Two-point average, or None unless both points are present."""
    if current is None or prior is None:
        return None
    return (current + prior) / 2.0


def _less(value: float | None, subtracted: float | None) -> float | None:
    """Difference of two figures, or None unless both are present."""
    if value is None or subtracted is None:
        return None
    return value - subtracted


def _piotroski_series(
    rows: Sequence[tuple[IncomePeriod, BalancePeriod | None, CashflowPeriod | None]],
    financial: bool,
) -> list[PiotroskiPoint]:
    """Scores every year that has the three balance sheets it needs."""
    out: list[PiotroskiPoint] = []
    for position in range(2, len(rows)):
        scored = _piotroski_at(rows, position, financial)
        out.append(
            PiotroskiPoint(
                period=scored.period,
                score=scored.score,
                computable=scored.computable,
            )
        )
    return out


# --- Altman Z-Score -------------------------------------------------------


def _altman_components(
    income: IncomePeriod,
    balance: BalancePeriod,
    equity_value: float | None,
    equity_label: str,
    weights: dict[str, float],
) -> list[AltmanComponent]:
    """Builds the five weighted terms of a Z-Score.

    Total assets and total liabilities are the denominators throughout, and
    both are taken through the positive-only guard: a negative or absent
    asset base does not produce a low score, it produces no score.
    """
    assets = balance.total_assets
    liabilities = balance.total_liabilities
    return [
        AltmanComponent(
            key="working_capital",
            name="Working capital / assets",
            definition="(Current assets less current liabilities) / total assets",
            ratio=fmt.pos_div(balance.working_capital, assets),
            weight=weights["working_capital"],
        ),
        AltmanComponent(
            key="retained_earnings",
            name="Retained earnings / assets",
            definition="Accumulated retained earnings / total assets",
            ratio=fmt.pos_div(balance.retained_earnings, assets),
            weight=weights["retained_earnings"],
        ),
        AltmanComponent(
            key="ebit",
            name="EBIT / assets",
            definition="Operating profit / total assets",
            ratio=fmt.pos_div(income.ebit, assets),
            weight=weights["ebit"],
        ),
        AltmanComponent(
            key="equity_value",
            name=f"{equity_label} / liabilities",
            definition=f"{equity_label} / total liabilities",
            ratio=fmt.pos_div(equity_value, liabilities),
            weight=weights["equity_value"],
        ),
        AltmanComponent(
            key="sales",
            name="Revenue / assets",
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
        return Altman(
            withheld_reason=(
                "The Z-Score is withheld for financials. Altman's model was fitted "
                "on manufacturers and two of its five terms do not describe a "
                "lender: working capital is not a meaningful concept where customer "
                "deposits sit in current liabilities, and revenue over assets "
                "measures balance-sheet turnover rather than operating efficiency. "
                "A number can be produced; it would not mean anything."
            )
        )

    balance = snap.latest_balance
    if balance is None:
        return Altman(withheld_reason="No balance sheet reported.")
    income = next((y for y in reversed(snap.years) if y.period == balance.period), None)
    if income is None:
        return Altman(
            withheld_reason=(
                "No income statement matches the latest balance-sheet period."
            )
        )

    equity_value = snap.market_cap_cr or snap.implied_market_cap_cr
    basis = (
        "Market capitalisation"
        if snap.market_cap_cr
        else "Implied market capitalisation"
    )
    return Altman(
        period=balance.period,
        model="Altman Z (1968), listed-company model",
        equity_basis=basis,
        components=_altman_components(
            income, balance, equity_value, basis, ALTMAN_PUBLIC_WEIGHTS
        ),
        safe_above=ALTMAN_PUBLIC_SAFE,
        distress_below=ALTMAN_PUBLIC_DISTRESS,
    )


def _altman_series(snap: CompanySnapshot, financial: bool) -> list[AltmanPoint]:
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
    out: list[AltmanPoint] = []
    for income in snap.years:
        balance = balance_by_period.get(income.period)
        if balance is None:
            continue
        score = Altman(
            period=income.period,
            components=_altman_components(
                income, balance, balance.equity, "Book equity", ALTMAN_PRIVATE_WEIGHTS
            ),
            safe_above=ALTMAN_PRIVATE_SAFE,
            distress_below=ALTMAN_PRIVATE_DISTRESS,
        )
        out.append(
            AltmanPoint(period=income.period, score=score.score, zone=score.zone)
        )
    return out


# --- capital allocation: sources and uses ---------------------------------


class CashTotals(NamedTuple):
    """Cash-flow lines accumulated across the window.

    Capex and dividends are the provider's positive outflow magnitudes, while
    the investing and financing figures are signed. The two `other_` members
    are what remains of each signed section once its named magnitude is added
    back, and are residuals rather than reported lines.
    """

    cfo: float
    capex: float
    dividends: float
    investing: float
    financing: float
    other_investing: float
    other_financing: float
    net_cash_change: float


def _cash_totals(years: Sequence[CashflowPeriod]) -> CashTotals:
    """Accumulates the cash-flow statement across the window."""
    cfo = sum(c.cfo for c in years)
    capex = sum(c.capex for c in years)
    dividends = sum(c.dividends_paid for c in years)
    investing = sum(c.cash_from_investing for c in years)
    financing = sum(c.cash_from_financing for c in years)
    return CashTotals(
        cfo=cfo,
        capex=capex,
        dividends=dividends,
        investing=investing,
        financing=financing,
        # Adding a positive magnitude back to its signed section total
        # leaves everything else in that section.
        other_investing=investing + capex,
        other_financing=financing + dividends,
        net_cash_change=cfo + investing + financing,
    )


def _flow_rows(totals: CashTotals) -> list[tuple[float, str, str, str]]:
    """The six net flows, each with the label it takes in either column.

    Held as data rather than as a run of calls so that the wording sits in one
    readable table: every line has an inflow name, an outflow name, and the
    definition printed beneath the exhibit.
    """
    return [
        (
            totals.cfo,
            "Operating cash flow",
            "Operating cash outflow",
            "Cash generated by operations, before capital expenditure",
        ),
        (
            -totals.capex,
            "Capital expenditure recovered",
            "Capital expenditure",
            "Purchases of property, plant, equipment and intangibles",
        ),
        (
            totals.other_investing,
            "Net disposals and investment maturities",
            "Net acquisitions and investments",
            "Investing cash flow other than capital expenditure; a residual",
        ),
        (
            -totals.dividends,
            "Dividends received",
            "Dividends paid",
            "Cash dividends paid to shareholders",
        ),
        # Deliberately neutral about what the residual contains. Calling the
        # outflow "net debt repaid" would have been wrong for one large IT
        # company, where the line was 44,374 crore of outflow while total debt
        # ROSE by 10,567: the money went to buybacks and lease payments, not
        # to retiring debt. The statement does not separate them, so the label
        # must not pretend it does.
        (
            totals.other_financing,
            "Net financing raised, other than dividends",
            "Net financing returned, other than dividends",
            "Financing cash flow other than dividends; a residual covering debt "
            "drawn and repaid, buybacks and lease payments together",
        ),
        (
            -totals.net_cash_change,
            "Cash and equivalents drawn down",
            "Cash and equivalents built up",
            "The closing balance the other lines leave behind",
        ),
    ]


def _split_flows(
    rows: Sequence[tuple[float, str, str, str]],
) -> tuple[list[FlowItem], list[FlowItem]]:
    """Files each net flow into whichever column its sign belongs to.

    A nil flow is dropped rather than printed as a zero row: the exhibit is a
    statement of what happened, and nothing happening is not a line.
    """
    sources: list[FlowItem] = []
    uses: list[FlowItem] = []
    for amount, source_label, use_label, definition in rows:
        if amount == 0:
            continue
        item = FlowItem(
            label=source_label if amount > 0 else use_label,
            amount=amount,
            definition=definition,
        )
        (sources if amount > 0 else uses).append(item)
    return sources, uses


def _balance_corroboration(
    snap: CompanySnapshot,
    years: Sequence[CashflowPeriod],
) -> tuple[float | None, float | None]:
    """Movement in debt and share count across the same window.

    The cash-flow residuals cannot say whether financing went to debt or to
    buybacks; the balance sheet can corroborate, which is why both endpoints
    are read here rather than inferred from the flows.

    Returns:
        Change in total debt, and change in share count as a percentage.

    """
    window = {c.period for c in years}
    balances = [b for b in snap.balance if b.period in window]
    if len(balances) < 2:
        return None, None

    first, last = balances[0], balances[-1]
    debt_change = (
        None if (first.debt is None or last.debt is None) else last.debt - first.debt
    )
    share_change = (
        fmt.pos_margin(last.shares_cr - first.shares_cr, first.shares_cr)
        if (first.shares_cr and last.shares_cr)
        else None
    )
    return debt_change, share_change


def _sources_and_uses(snap: CompanySnapshot, window: int) -> SourcesAndUses:
    """Aggregates the cash-flow statement into a sources-and-uses statement.

    Rests on the statement's own articulation -- operating plus investing plus
    financing equals the change in cash -- so the two columns balance by
    identity, which `selfcheck` asserts for every report.

    Two lines are residuals, labelled as such on the page: investing beyond
    capex, and financing beyond dividends. Neither decomposes further, because
    the provider publishes no debt-raised, debt-repaid or buyback line.

    Args:
        snap: Populated company snapshot.
        window: Maximum number of fiscal years to accumulate.

    Returns:
        A SourcesAndUses record. Years missing any of the five inputs are
        excluded from the accumulation rather than treated as nil, and the
        window actually used is recorded on the result.

    """
    usable = [
        c
        for c in snap.cashflow
        if None
        not in (
            c.cfo,
            c.capex,
            c.fcf,
            c.dividends_paid,
            c.cash_from_investing,
            c.cash_from_financing,
        )
    ][-window:]
    if not usable:
        return SourcesAndUses(
            notes=[
                "The provider reports no cash flow statement for this company, so "
                "capital allocation cannot be traced."
            ]
        )

    result = SourcesAndUses(
        from_period=usable[0].period,
        to_period=usable[-1].period,
        years=len(usable),
    )
    result.sources, result.uses = _split_flows(_flow_rows(_cash_totals(usable)))
    result.debt_change, result.share_change_pct = _balance_corroboration(snap, usable)

    if len(usable) < min(window, len(snap.cashflow)):
        result.notes.append(
            f"Accumulated over the {len(usable)} years with a complete cash flow "
            "statement; earlier years are missing one or more lines."
        )
    return result


# --- capital allocation: the reinvestment identity ------------------------


def _reinvestment_year(
    income: IncomePeriod,
    prior: IncomePeriod,
    balance: BalancePeriod,
    prior_balance: BalancePeriod,
    cash: CashflowPeriod,
    metrics: AnnualMetrics,
) -> ReinvestmentYear:
    """Measures one year: capital put in, and the growth that implies.

    Reinvestment is capital expenditure net of depreciation plus the increase
    in non-cash working capital -- what the business put in beyond replacing
    what wore out. Depreciation stands in for maintenance capital, which is
    the standard simplification and is stated on the page.
    """
    delta_working_capital = _less(
        _non_cash_working_capital(balance), _non_cash_working_capital(prior_balance)
    )
    net_capex = _less(cash.capex, income.depreciation)
    reinvestment = (
        None
        if (net_capex is None or delta_working_capital is None)
        else net_capex + delta_working_capital
    )
    rate = fmt.pos_margin(reinvestment, metrics.nopat)

    return ReinvestmentYear(
        period=income.period,
        nopat=metrics.nopat,
        capex=cash.capex,
        depreciation=income.depreciation,
        net_capex=net_capex,
        delta_working_capital=delta_working_capital,
        reinvestment=reinvestment,
        reinvestment_rate=rate,
        roic=metrics.roic,
        implied_growth=(
            None
            if (rate is None or metrics.roic is None)
            else rate / 100.0 * metrics.roic
        ),
        revenue_growth=fmt.growth(income.revenue, prior.revenue),
    )


def _reinvestment_years(
    snap: CompanySnapshot,
    derived: DerivedAnalytics,
) -> list[ReinvestmentYear]:
    """Measures every year that has the statements the identity needs.

    A year is skipped rather than part-measured: the working-capital movement
    needs the prior balance sheet, so the first year of any series cannot be
    measured at all.
    """
    metrics_by_period: dict[str, AnnualMetrics] = {m.period: m for m in derived.annual}
    balance_by_period = {b.period: b for b in snap.balance}
    cash_by_period = {c.period: c for c in snap.cashflow}

    years: list[ReinvestmentYear] = []
    for position, income in enumerate(snap.years):
        if position == 0:
            continue
        prior = snap.years[position - 1]
        balance = balance_by_period.get(income.period)
        prior_balance = balance_by_period.get(prior.period)
        cash = cash_by_period.get(income.period)
        metrics = metrics_by_period.get(income.period)
        if balance is None or prior_balance is None or cash is None or metrics is None:
            continue
        years.append(
            _reinvestment_year(income, prior, balance, prior_balance, cash, metrics)
        )
    return years


def _reinvestment_window(
    years: Sequence[ReinvestmentYear],
) -> tuple[float | None, float | None, float | None]:
    """Totals the window: reinvestment, NOPAT, and the rate between them.

    The rate is aggregated, not averaged. A mean of annual rates lets one
    restructuring year dominate the answer: a single working-capital swing
    can produce a 234% annual rate, which drags a seven-year mean far from
    anything the company actually did. Summing both sides first weights each
    year by its own size, which is what the identity is asking about.

    Returns:
        Total reinvestment, total NOPAT and the aggregate rate; all three are
        None unless every year in the window reports both figures, since a
        partial total would be read as a complete one.

    """
    reinvestments = [y.reinvestment for y in years if y.reinvestment is not None]
    nopats = [y.nopat for y in years if y.nopat is not None]
    if not reinvestments or len(reinvestments) != len(nopats):
        return None, None, None
    total_reinvestment, total_nopat = sum(reinvestments), sum(nopats)
    return (
        total_reinvestment,
        total_nopat,
        fmt.pos_margin(total_reinvestment, total_nopat),
    )


def _delivered_growth(
    snap: CompanySnapshot,
    years: Sequence[ReinvestmentYear],
) -> tuple[float | None, float | None]:
    """Revenue and EBIT CAGR across the same window the identity covers.

    Measured over the identity's own window so that the two sides of the
    comparison describe one period rather than two.
    """
    window = {y.period for y in years}
    income = [y for y in snap.years if y.period in window]
    if len(income) < 2:
        return None, None
    spans = len(income) - 1
    return (
        _cagr(income[0].revenue, income[-1].revenue, spans),
        _cagr(income[0].ebit, income[-1].ebit, spans),
    )


def _reinvestment_notes(
    result: Reinvestment,
    years: Sequence[ReinvestmentYear],
) -> list[str]:
    """The caveats that keep the exhibit readable as evidence.

    Two of the three explain a gap the reader would otherwise take as a
    contradiction, which is the difference between an exhibit that informs
    and one that looks broken.
    """
    notes: list[str] = []

    withheld = [
        y.period
        for y in years
        if y.reinvestment is not None and y.reinvestment_rate is None
    ]
    if withheld:
        notes.append(
            "The reinvestment rate is withheld for "
            f"{', '.join(fmt.period_label(p) for p in withheld)}: net operating profit "
            "after tax was nil or negative, and a rate against that base "
            "carries no meaning."
        )

    # The identity only sees investment that lands on the balance sheet. A
    # business that grows by hiring, or by spending on research and brand,
    # expenses all of it, so reinvestment reads at or below zero while
    # revenue compounds.
    net_capex = [y.net_capex for y in years if y.net_capex is not None]
    below_depreciation = sum(1 for v in net_capex if v < 0)
    if len(net_capex) >= 3 and below_depreciation > len(net_capex) / 2.0:
        notes.append(
            "Capital expenditure ran below depreciation in most years of the "
            "window, so measured reinvestment is negative and the implied "
            "growth rate falls below the growth delivered. That gap is a "
            "property of the measure rather than a contradiction: this "
            "identity counts only investment that is capitalised onto the "
            "balance sheet, and a business that grows through headcount, "
            "research or brand expenses that spending as it goes. Read the "
            "gap as the share of growth this framework cannot attribute."
        )
    elif result.growth_gap is not None and abs(result.growth_gap) > 5.0:
        notes.append(
            "Delivered revenue growth differs from the growth reinvestment "
            f"implies by {fmt.signed_pct(result.growth_gap)}. A positive gap points to "
            "growth from sources this "
            "identity does not capture, such as pricing, acquisitions "
            "accounted for outside capital expenditure, or expensed "
            "investment; a negative gap means capital went in without "
            "commensurate growth coming out."
        )
    return notes


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
        return Reinvestment(
            withheld_reason=(
                "The reinvestment identity is withheld for financials. It measures "
                "growth bought with fixed assets and working capital, and a lender "
                "grows by deploying its balance sheet instead: capital expenditure "
                "is incidental to the business and working capital is not a "
                "meaningful concept where deposits are a current liability."
            )
        )

    years = _reinvestment_years(snap, derived)[-window:]
    if not years:
        return Reinvestment(
            withheld_reason=(
                "Two consecutive years of balance sheet and cash flow are needed "
                "to measure reinvestment, and the provider reports fewer."
            )
        )

    result = Reinvestment(
        years=years,
        window_from=years[0].period,
        window_to=years[-1].period,
        average_roic=_mean([y.roic for y in years]),
    )

    (
        result.total_reinvestment,
        result.total_nopat,
        result.aggregate_reinvestment_rate,
    ) = _reinvestment_window(years)
    if (
        result.aggregate_reinvestment_rate is not None
        and result.average_roic is not None
    ):
        result.implied_growth = (
            result.aggregate_reinvestment_rate / 100.0 * result.average_roic
        )

    result.actual_revenue_cagr, result.actual_ebit_cagr = _delivered_growth(snap, years)
    result.notes.extend(_reinvestment_notes(result, years))
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

    rows: list[DupontYear] = []
    for income in periods:
        balance = balance_by_period.get(income.period)
        # The trailing row has no balance sheet of its own; the latest
        # reported one is its correct pairing, as elsewhere in the report.
        if balance is None and income is snap.ttm and snap.balance:
            balance = snap.balance[-1]
        if balance is None:
            continue
        rows.append(
            DupontYear(
                period=income.period,
                tax_burden=fmt.pos_div(income.pat, income.pbt),
                interest_burden=fmt.pos_div(income.pbt, income.ebit),
                operating_margin=fmt.pos_margin(income.ebit, income.revenue),
                asset_turnover=fmt.pos_div(income.revenue, balance.total_assets),
                equity_multiplier=fmt.pos_div(balance.total_assets, balance.equity),
                roe_direct=fmt.pos_margin(income.pat, balance.equity),
            )
        )

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
        match = next(
            (
                r
                for r in rows
                if r.period.strip().upper() == result.provider_period.strip().upper()
            ),
            None,
        )
        if (
            match is not None
            and match.roe_product is not None
            and result.provider_roe is not None
        ):
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
        piotroski = Piotroski(
            withheld=[
                "The F-Score needs three consecutive years of statements, because "
                "its change tests are scaled by opening assets. The provider "
                f"reports {len(rows)}."
            ]
        )
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
            f"The F-Score is reported out of {piotroski.computable} evaluated signals "
            "rather than "
            "nine and is therefore not comparable with a standard F-Score."
        )
    return composites
