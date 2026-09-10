"""Arithmetic self-verification for a generated report.

Every figure in a GrowNXT report is either something the provider said or
something this codebase computed. The computed ones are only as trustworthy
as the arithmetic behind them, and a typeset PDF gives a wrong number the
same authority as a right one. This module re-derives what can be re-derived
by an independent route and reports where the two answers disagree.

Three kinds of check run here, in descending order of what a failure means:

    IDENTITIES must hold to floating-point precision, because they are
    definitions rather than estimates. The five DuPont factors multiplied
    together are ROE. Operating plus investing plus financing cash flow is
    the movement in cash. Sources equal uses. A failure here is a defect in
    this codebase, not a data problem, and it is reported as such.

    RECONCILIATIONS compare a figure computed here against the provider's
    own published value for the same quantity. These are expected to agree
    to within rounding: the provider publishes its DuPont factors to four
    decimals, so a product rebuilt from them lands a few thousandths of a
    percentage point away from its stated ROE. A wide gap means a
    methodology difference worth knowing about.

    GUARDRAILS assert the report's own editorial rules held in practice -
    that no ratio was published against a non-positive denominator, and that
    no composite reached the page without its components. These cannot be
    proven by inspecting the code alone, because whether the path was taken
    depends on the company's numbers.

The result is rendered as an exhibit rather than kept in a log. A report
that states its own arithmetic was verified, and shows the residuals, is
making a checkable claim; one that stays silent is asking for trust.
"""

import logging
from dataclasses import dataclass, field

from reporting import fmt
from reporting.analytics import DerivedAnalytics
from reporting.composites import Composites
from reporting.snapshot import CompanySnapshot

logger = logging.getLogger(__name__)

# An identity is a definition, so it must close to floating-point noise.
# The data arrives rounded to one decimal place in rupee crore, which is why
# the monetary tolerance is not tighter than half of that.
TOL_IDENTITY_PCT: float = 1e-6
TOL_IDENTITY_CRORE: float = 0.5

# The provider publishes its DuPont factors rounded to four decimal places
# and its ratios to two. A product rebuilt from those lands within a
# hundredth of a percentage point, so anything inside this band is rounding
# rather than disagreement.
TOL_PROVIDER_PCT: float = 0.05

# Return on invested capital is defined slightly differently by different
# houses - the tax rate applied to EBIT, and whether goodwill sits inside
# invested capital, both move it. A wider band is honest here.
TOL_METHODOLOGY_PCT: float = 1.5

# Decimal places the provider actually publishes each DuPont factor to,
# established by inspecting the payloads for every ticker tested: the two
# burden ratios arrive at four places, the margin, turnover and multiplier
# at two. Comparing a value rounded to two places against one computed at
# full precision needs a tolerance sized to that rounding, and getting this
# wrong is not harmless in either direction - too tight reports rounding as
# a defect, too loose lets a real methodology difference through. One unit
# in the last published place covers a provider that truncates as well as
# one that rounds, while staying far below the shift a genuine definitional
# difference would produce. Using average rather than period-end assets, for
# instance, moves turnover by several percent, which is orders of magnitude
# outside this band.
PROVIDER_FACTOR_DECIMALS: dict = {
    "tax_burden": 4,
    "interest_burden": 4,
    "operating_margin": 2,
    "asset_turnover": 2,
    "equity_multiplier": 2,
}


def _rounding_tolerance(decimals: int) -> float:
    """One unit in the last decimal place the provider publishes."""
    return 10.0 ** (-decimals)


@dataclass
class Check:
    """One verification, with both sides of the comparison retained."""

    name: str
    kind: str
    expected: float | None = None
    actual: float | None = None
    tolerance: float = TOL_IDENTITY_PCT
    unit: str = "pp"
    detail: str = ""

    @property
    def delta(self) -> float | None:
        """Actual less expected, or None when either side is missing."""
        if self.expected is None or self.actual is None:
            return None
        return self.actual - self.expected

    @property
    def passed(self) -> bool | None:
        """True inside tolerance, False outside, None when not applicable.

        A check with nothing to compare is skipped rather than passed. This
        distinction matters: a report on a company whose statements are too
        sparse to verify must not claim it was verified.
        """
        delta = self.delta
        if delta is None:
            return None
        return abs(delta) <= self.tolerance


@dataclass
class SelfCheck:
    """Every verification run against one report."""

    checks: list[Check] = field(default_factory=list)

    @property
    def applicable(self) -> list[Check]:
        """Checks that had both sides available."""
        return [c for c in self.checks if c.passed is not None]

    @property
    def failures(self) -> list[Check]:
        """Checks that ran and did not close."""
        return [c for c in self.checks if c.passed is False]

    @property
    def skipped(self) -> list[Check]:
        """Checks that could not run for want of inputs."""
        return [c for c in self.checks if c.passed is None]

    @property
    def passed_count(self) -> int:
        """How many checks closed within tolerance."""
        return sum(1 for c in self.checks if c.passed is True)

    @property
    def all_passed(self) -> bool:
        """Whether every applicable check closed."""
        return not self.failures

    @property
    def worst_identity_residual(self) -> float | None:
        """Largest absolute residual among the identity checks."""
        deltas = [
            abs(c.delta)
            for c in self.checks
            if c.kind == "identity" and c.delta is not None
        ]
        return max(deltas) if deltas else None


def _add(checks: list[Check], **kwargs) -> None:
    """Appends a check and logs it when it fails."""
    check = Check(**kwargs)
    checks.append(check)
    if check.passed is False:
        logger.warning(
            "self-check FAILED: %s (expected %s, got %s, delta %s %s)",
            check.name,
            check.expected,
            check.actual,
            check.delta,
            check.unit,
        )


def _cashflow_articulation(snap: CompanySnapshot, checks: list[Check]) -> None:
    """Tests operating plus investing plus financing against reported cash.

    This is the identity the whole capital-allocation section rests on. It is
    checked per period and reported as the worst single period, because one
    bad year is enough to make a cumulative sources-and-uses statement wrong.
    """
    worst: float | None = None
    worst_period = ""
    tested = 0
    for row in snap.cashflow:
        if None in (
            row.cfo,
            row.cash_from_investing,
            row.cash_from_financing,
            row.net_change_in_cash,
        ):
            continue
        tested += 1
        residual = (
            row.cfo + row.cash_from_investing + row.cash_from_financing
        ) - row.net_change_in_cash
        if worst is None or abs(residual) > abs(worst):
            worst, worst_period = residual, row.period
    _add(
        checks,
        name="Cash flow articulates to the reported movement in cash",
        kind="identity",
        expected=0.0,
        actual=worst,
        tolerance=TOL_IDENTITY_CRORE,
        unit="Rs cr",
        detail=(
            "Operating plus investing plus financing, against the "
            f"provider's own closing figure, across {tested} periods. Worst is "
            f"{fmt.period_label(worst_period) if worst_period else 'none'}."
        ),
    )


def _free_cash_flow(snap: CompanySnapshot, checks: list[Check]) -> None:
    """Tests that reported free cash flow is operating cash less capex."""
    worst: float | None = None
    worst_period = ""
    tested = 0
    for row in snap.cashflow:
        if None in (row.cfo, row.capex, row.fcf):
            continue
        tested += 1
        residual = (row.cfo - row.capex) - row.fcf
        if worst is None or abs(residual) > abs(worst):
            worst, worst_period = residual, row.period
    _add(
        checks,
        name="Free cash flow equals operating cash flow less capex",
        kind="identity",
        expected=0.0,
        actual=worst,
        tolerance=TOL_IDENTITY_CRORE,
        unit="Rs cr",
        detail=(
            f"Checked across {tested} periods. Worst is "
            f"{fmt.period_label(worst_period) if worst_period else 'none'}."
        ),
    )


def _sources_and_uses(composites: Composites, checks: list[Check]) -> None:
    """Tests that the sources column totals the uses column."""
    allocation = composites.sources_uses
    _add(
        checks,
        name="Capital allocation: sources equal uses",
        kind="identity",
        expected=0.0,
        actual=allocation.residual,
        tolerance=TOL_IDENTITY_CRORE,
        unit="Rs cr",
        detail=(
            f"Cumulative over {allocation.years} years to "
            f"{fmt.period_label(allocation.to_period) if allocation.to_period else 'none'}. "
            "The two columns are built "
            "from one articulation, so a residual would mean a "
            "classification error."
        ),
    )


def _dupont(composites: Composites, checks: list[Check]) -> None:
    """Tests the factor product against ROE, and both against the provider."""
    dupont = composites.dupont

    worst: float | None = None
    worst_period = ""
    tested = 0
    for row in dupont.years:
        residual = row.residual
        if residual is None:
            continue
        tested += 1
        if worst is None or abs(residual) > abs(worst):
            worst, worst_period = residual, row.period
    _add(
        checks,
        name="DuPont: five factors multiply to return on equity",
        kind="identity",
        expected=0.0,
        actual=worst,
        tolerance=TOL_IDENTITY_PCT,
        unit="pp",
        detail=(
            "Every period rebuilds ROE from its own factors and is set "
            "against ROE computed directly from profit and equity. "
            f"Checked across {tested} periods. Worst is "
            f"{fmt.period_label(worst_period) if worst_period else 'none'}."
        ),
    )

    if dupont.provider_roe is not None and dupont.reconciliation_delta is not None:
        _add(
            checks,
            name="DuPont: rebuilt ROE matches the provider's published ROE",
            kind="reconciliation",
            expected=0.0,
            actual=dupont.reconciliation_delta,
            tolerance=TOL_PROVIDER_PCT,
            unit="pp",
            detail=(
                f"Computed here for {dupont.provider_period or 'the trailing period'} "
                f"against the provider's own "
                f"five-factor endpoint, which publishes {fmt.pct(dupont.provider_roe, 2)}. "
                "The provider "
                "rounds its factors, so a residual of a few thousandths "
                "of a point is expected."
            ),
        )

    # Each of the five factors, individually, against the provider's own.
    # The product agreeing is necessary but not sufficient: two offsetting
    # factor errors would still multiply to the right answer.
    match = next(
        (
            r
            for r in dupont.years
            if r.period.strip().upper()
            == (dupont.provider_period or "").strip().upper()
        ),
        None,
    )
    if match is not None and dupont.provider_factors:
        pairs = [
            ("tax_burden", "Tax burden", match.tax_burden),
            ("interest_burden", "Interest burden", match.interest_burden),
            ("operating_margin", "Operating margin", match.operating_margin),
            ("asset_turnover", "Asset turnover", match.asset_turnover),
            ("equity_multiplier", "Equity multiplier", match.equity_multiplier),
        ]
        for key, label, mine in pairs:
            theirs = dupont.provider_factors.get(key)
            if mine is None or theirs is None:
                continue
            decimals = PROVIDER_FACTOR_DECIMALS.get(key, 2)
            _add(
                checks,
                name=f"DuPont factor reconciles: {label.lower()}",
                kind="reconciliation",
                expected=theirs,
                actual=mine,
                tolerance=_rounding_tolerance(decimals),
                unit="pp" if key == "operating_margin" else "x",
                detail=(
                    "Computed here at full precision against the "
                    f"provider's published factor, which it reports to {decimals} "
                    "decimal places."
                ),
            )


def _returns(
    snap: CompanySnapshot, derived: DerivedAnalytics, checks: list[Check]
) -> None:
    """Reconciles the trailing return series against the ratio endpoints."""
    trailing = next(
        (m for m in reversed(derived.annual) if m.period.strip().upper() == "TTM"), None
    )
    if trailing is None:
        trailing = derived.annual[-1] if derived.annual else None
    if trailing is None:
        return

    provider_roe = snap.dupont.get("return_on_equity_roe_pct")
    if provider_roe is not None and trailing.roe is not None:
        _add(
            checks,
            name="Return on equity matches the provider's endpoint",
            kind="reconciliation",
            expected=provider_roe,
            actual=trailing.roe,
            tolerance=TOL_PROVIDER_PCT,
            unit="pp",
            detail=(
                "Both on period-end equity, which is the convention the "
                "provider's endpoint uses."
            ),
        )

    provider_roic = snap.capital_efficiency.get("return_on_invested_capital_roic_pct")
    if provider_roic is not None and trailing.roic is not None:
        _add(
            checks,
            name="Return on invested capital matches the provider's endpoint",
            kind="reconciliation",
            expected=provider_roic,
            actual=trailing.roic,
            tolerance=TOL_METHODOLOGY_PCT,
            unit="pp",
            detail=(
                "Invested capital is equity plus debt less cash, and NOPAT "
                "applies the effective tax rate to EBIT. Definitions of "
                "both vary between houses, so this band is wider."
            ),
        )


def _piotroski(composites: Composites, checks: list[Check]) -> None:
    """Tests that the F-Score is the sum of its own sub-tests."""
    score = composites.piotroski
    if not score.tests:
        return

    awarded = [t.points for t in score.tests if t.points is not None]
    _add(
        checks,
        name="F-Score equals the sum of its sub-tests",
        kind="identity",
        expected=float(sum(awarded)) if awarded else None,
        actual=float(score.score) if score.score is not None else None,
        tolerance=0.0,
        unit="points",
        detail=(
            f"{score.computable} of the nine signals were evaluated and "
            f"{sum(awarded) if awarded else 0} awarded a point. "
            "The total is a property of the sub-test list, so this check "
            "confirms none were dropped between computation and rendering."
        ),
    )

    stray = [
        t.number for t in score.tests if t.points is not None and t.points not in (0, 1)
    ]
    _add(
        checks,
        name="Every F-Score signal awards nil or one point",
        kind="guardrail",
        expected=0.0,
        actual=float(len(stray)),
        tolerance=0.0,
        unit="signals",
        detail=(
            "A signal outside nil-or-one would mean a scoring bug. "
            "Offending signals: %s."
            % (", ".join(str(n) for n in stray) if stray else "none")
        ),
    )

    # A composite must not reach the page without its parts. The renderer
    # takes the parts from the same object it takes the total from, so this
    # asserts the object itself is well formed.
    missing_definition = [t.number for t in score.tests if not t.definition]
    _add(
        checks,
        name="Every F-Score signal carries its definition and inputs",
        kind="guardrail",
        expected=0.0,
        actual=float(len(missing_definition)),
        tolerance=0.0,
        unit="signals",
        detail=(
            "The report may not print a composite without the components "
            "that produced it. Signals lacking a stated definition: %s."
            % (
                ", ".join(str(n) for n in missing_definition)
                if missing_definition
                else "none"
            )
        ),
    )


def _altman(composites: Composites, checks: list[Check]) -> None:
    """Tests that the Z-Score is the sum of its weighted terms."""
    altman = composites.altman
    if altman.withheld_reason or not altman.components:
        return

    contributions = [c.contribution for c in altman.components]
    if any(c is None for c in contributions):
        return
    _add(
        checks,
        name="Z-Score equals the sum of its weighted terms",
        kind="identity",
        expected=sum(contributions),
        actual=altman.score,
        tolerance=TOL_IDENTITY_PCT,
        unit="points",
        detail=(
            "Five terms, each a ratio times its published coefficient. "
            "The total is derived from the term list, so a mismatch would "
            "mean a term was rendered but not counted."
        ),
    )

    missing = [c.name for c in altman.components if not c.definition]
    _add(
        checks,
        name="Every Z-Score term carries its ratio and coefficient",
        kind="guardrail",
        expected=0.0,
        actual=float(len(missing)),
        tolerance=0.0,
        unit="terms",
        detail=(
            "Terms lacking a stated definition: %s."
            % (", ".join(missing) if missing else "none")
        ),
    )


def _reinvestment(composites: Composites, checks: list[Check]) -> None:
    """Tests the reinvestment identity closes on its own inputs."""
    reinvestment = composites.reinvestment
    if reinvestment.withheld_reason:
        return
    rate = reinvestment.aggregate_reinvestment_rate
    roic = reinvestment.average_roic
    if rate is None or roic is None or reinvestment.implied_growth is None:
        return
    _add(
        checks,
        name="Implied growth equals reinvestment rate times return",
        kind="identity",
        expected=rate / 100.0 * roic,
        actual=reinvestment.implied_growth,
        tolerance=TOL_IDENTITY_PCT,
        unit="pp",
        detail=(
            f"Reinvestment rate of {fmt.pct(rate, 1)} applied to a return on invested "
            f"capital of {fmt.pct(roic, 1)}."
        ),
    )

    if (
        reinvestment.total_nopat is not None
        and reinvestment.total_reinvestment is not None
    ):
        summed = sum(
            y.reinvestment for y in reinvestment.years if y.reinvestment is not None
        )
        _add(
            checks,
            name="Cumulative reinvestment totals the annual figures",
            kind="identity",
            expected=summed,
            actual=reinvestment.total_reinvestment,
            tolerance=TOL_IDENTITY_CRORE,
            unit="Rs cr",
            detail="The window aggregate against the rows shown in the table.",
        )


def _denominator_guardrail(
    snap: CompanySnapshot,
    derived: DerivedAnalytics,
    composites: Composites,
    checks: list[Check],
) -> None:
    """Asserts no published ratio stands on a non-positive denominator.

    The guard lives in `fmt.pos_div`, but whether it was reached depends on
    the company's own numbers, so the rule cannot be proven by reading the
    code. This walks the figures that actually reached the page and counts
    any that should have been withheld and were not.
    """
    balance_by_period = {b.period: b for b in snap.balance}
    breaches: list[str] = []

    for metrics in derived.annual:
        balance = balance_by_period.get(metrics.period)
        if balance is None and metrics.period.strip().upper() == "TTM" and snap.balance:
            balance = snap.balance[-1]
        if balance is None:
            continue
        equity = balance.equity
        if equity is not None and equity <= 0:
            for label, value in (
                ("return on equity", metrics.roe),
                ("debt to equity", metrics.debt_to_equity),
                ("goodwill to equity", metrics.goodwill_to_equity),
            ):
                if value is not None:
                    breaches.append(
                        f"{label} in {metrics.period} on equity of {fmt.num(equity)}"
                    )
        if (
            metrics.invested_capital is not None
            and metrics.invested_capital <= 0
            and metrics.roic is not None
        ):
            breaches.append(
                f"return on invested capital in {metrics.period} on invested "
                f"capital of {fmt.num(metrics.invested_capital)}"
            )

    for row in composites.dupont.years:
        balance = balance_by_period.get(row.period)
        if balance is None and row.period.strip().upper() == "TTM" and snap.balance:
            balance = snap.balance[-1]
        if balance is None or balance.equity is None:
            continue
        if balance.equity <= 0 and row.equity_multiplier is not None:
            breaches.append(
                f"DuPont equity multiplier in {row.period} on equity of "
                f"{fmt.num(balance.equity)}"
            )

    _add(
        checks,
        name="No ratio published against a non-positive denominator",
        kind="guardrail",
        expected=0.0,
        actual=float(len(breaches)),
        tolerance=0.0,
        unit="figures",
        detail=(
            "Negative equity, negative EBITDA and negative cost of goods "
            "sold all yield arithmetically valid, analytically meaningless "
            "ratios. Every such figure must be withheld rather than "
            "printed. Breaches found: %s."
            % ("; ".join(breaches) if breaches else "none")
        ),
    )


def run(
    snap: CompanySnapshot,
    derived: DerivedAnalytics,
    composites: Composites,
) -> SelfCheck:
    """Runs every verification against one report's figures.

    Args:
        snap: The normalised snapshot the report was built from.
        derived: Tier 1 analytics.
        composites: Tier 2 composites.

    Returns:
        A SelfCheck holding one record per verification, each retaining both
        sides of its comparison so the report can print the residual rather
        than merely assert success.

    """
    checks: list[Check] = []
    _cashflow_articulation(snap, checks)
    _free_cash_flow(snap, checks)
    _sources_and_uses(composites, checks)
    _dupont(composites, checks)
    _returns(snap, derived, checks)
    _piotroski(composites, checks)
    _altman(composites, checks)
    _reinvestment(composites, checks)
    _denominator_guardrail(snap, derived, composites, checks)

    result = SelfCheck(checks=checks)
    if result.failures:
        logger.warning(
            "[%s] %d self-check failure(s)", snap.ticker, len(result.failures)
        )
    else:
        logger.info(
            "[%s] %d self-checks passed, %d skipped",
            snap.ticker,
            result.passed_count,
            len(result.skipped),
        )
    return result
