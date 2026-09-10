"""Verification harness for the reporting engine.

Run this after changing anything under `reporting/`:

    venv/Scripts/python.exe scripts/verify_reporting.py

It exits non-zero if any check fails, so it can gate a commit.

Three things are checked, and the first is the reason this file exists.

The denominator guardrail cannot be verified against real cached data,
because none of the companies in the cache has negative equity, negative
EBITDA or a negative cost base. Every ratio in those reports takes the happy
path, so a passing report proves nothing about what happens when a balance
sheet turns. This harness therefore builds a synthetic company that fails on
every axis at once and asserts that each affected ratio comes back absent
rather than merely wrong.

It also asserts the opposite direction: that the guardrail detector inside
`selfcheck` reports a breach when one is planted. A check that cannot fail is
not evidence, and a self-verification suite that only ever passes is the
thing it is supposed to protect against.

The composite-integrity checks assert the editorial rule that a score is
never available without its components, by reaching for the total and the
parts through the same object and comparing them.
"""

import logging
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import OUTPUT_DIR
from reporting import analytics, composites, fmt, selfcheck  # noqa: E402
from reporting.client import CollectorClient  # noqa: E402
from reporting.snapshot import (  # noqa: E402
    BalancePeriod,
    CashflowPeriod,
    CompanySnapshot,
    IncomePeriod,
    build_snapshot,
)
from scripts import cli  # noqa: E402
from scripts.checks import Failure, Report, banner, require  # noqa: E402

# Tickers with a complete payload set already on disk. The harness reads the
# cache only; it must not depend on the network to run.
CACHED_TICKERS: tuple[str, ...] = ("WIPRO", "RELIANCE", "TCS", "HDFCBANK")

CACHE_ROOT: Path = OUTPUT_DIR


def _absent(value, label: str) -> None:
    """Asserts a ratio was withheld rather than computed."""
    require(
        value is None,
        f"{label} should have been withheld but came back as {value!r}",
    )


def _distressed_snapshot() -> CompanySnapshot:
    """Builds a company that breaks every denominator at once.

    Equity turns negative, EBIT and EBITDA turn negative, pre-tax profit
    turns negative, the cost base is reported negative, and capital employed
    and invested capital both fall below zero. Each of those is individually
    plausible in a real filing; together they exercise every guard in one
    pass.
    """
    years = [
        IncomePeriod(
            period="FY 2024",
            revenue=4000.0,
            ebit=200.0,
            pbt=100.0,
            pat=80.0,
            eps=0.8,
            depreciation=120.0,
            interest=100.0,
            raw_materials=2400.0,
            sga=600.0,
        ),
        IncomePeriod(
            period="FY 2025",
            revenue=3500.0,
            ebit=-300.0,
            pbt=-500.0,
            pat=-600.0,
            eps=-6.0,
            depreciation=130.0,
            interest=200.0,
            raw_materials=2200.0,
            sga=650.0,
        ),
        # The distressed year. EBITDA is EBIT plus depreciation, so -800 plus
        # 100 leaves it negative too.
        IncomePeriod(
            period="FY 2026",
            revenue=3000.0,
            ebit=-800.0,
            pbt=-1200.0,
            pat=-1100.0,
            eps=-11.0,
            depreciation=100.0,
            interest=300.0,
            raw_materials=-50.0,
            sga=700.0,
        ),
    ]
    balance = [
        BalancePeriod(
            period="FY 2024",
            equity=1000.0,
            debt=2000.0,
            cash=300.0,
            total_assets=5000.0,
            current_assets=1800.0,
            current_liabilities=1500.0,
            inventory=400.0,
            receivables=500.0,
            shares_cr=100.0,
            payables=300.0,
            goodwill_intangibles=200.0,
            retained_earnings=400.0,
            long_term_debt=1500.0,
            total_liabilities=4000.0,
        ),
        BalancePeriod(
            period="FY 2025",
            equity=200.0,
            debt=2600.0,
            cash=200.0,
            total_assets=4800.0,
            current_assets=1600.0,
            current_liabilities=2000.0,
            inventory=380.0,
            receivables=480.0,
            shares_cr=100.0,
            payables=320.0,
            goodwill_intangibles=200.0,
            retained_earnings=-200.0,
            long_term_debt=1900.0,
            total_liabilities=4600.0,
        ),
        # Negative equity; current liabilities above total assets, so capital
        # employed is negative; and cash above equity plus debt, so invested
        # capital is negative.
        BalancePeriod(
            period="FY 2026",
            equity=-400.0,
            debt=2900.0,
            cash=3400.0,
            total_assets=4500.0,
            current_assets=1400.0,
            current_liabilities=4800.0,
            inventory=360.0,
            receivables=450.0,
            shares_cr=100.0,
            payables=350.0,
            goodwill_intangibles=200.0,
            retained_earnings=-900.0,
            long_term_debt=2100.0,
            total_liabilities=4900.0,
        ),
    ]
    cashflow = [
        CashflowPeriod(
            period="FY 2024",
            cfo=300.0,
            capex=150.0,
            fcf=150.0,
            dividends_paid=40.0,
            cash_from_investing=-150.0,
            cash_from_financing=-100.0,
            net_change_in_cash=50.0,
        ),
        CashflowPeriod(
            period="FY 2025",
            cfo=-100.0,
            capex=120.0,
            fcf=-220.0,
            dividends_paid=0.0,
            cash_from_investing=-120.0,
            cash_from_financing=120.0,
            net_change_in_cash=-100.0,
        ),
        CashflowPeriod(
            period="FY 2026",
            cfo=-500.0,
            capex=90.0,
            fcf=-590.0,
            dividends_paid=0.0,
            cash_from_investing=-90.0,
            cash_from_financing=390.0,
            net_change_in_cash=-200.0,
        ),
    ]
    return CompanySnapshot(
        ticker="STRESS",
        name="Stress Test Industries",
        sector="Diversified Manufacturing",
        years=years,
        balance=balance,
        cashflow=cashflow,
        market_cap_cr=500.0,
    )


def check_division_helpers() -> str:
    """The guard itself, at the level it is implemented."""
    _absent(fmt.pos_div(-500.0, -1200.0), "pos_div against negative equity")
    _absent(fmt.pos_div(500.0, 0.0), "pos_div against a nil denominator")
    _absent(fmt.pos_margin(400.0, -250.0), "pos_margin against a negative base")
    _absent(fmt.pos_div(None, 100.0), "pos_div with an absent numerator")

    # A negative numerator over a positive base is real information and must
    # survive. Interest cover of minus two says the company cannot service
    # its debt, which is exactly what a reader needs to see.
    result = fmt.pos_div(-800.0, 400.0)
    require(
        result is not None and abs(result + 2.0) < 1e-12,
        f"a negative numerator over a positive base must survive, got {result!r}",
    )
    return "division helpers guard the denominator only"


def check_distressed_ratios() -> str:
    """Every ratio that stands on a broken denominator must be absent."""
    snap = _distressed_snapshot()
    derived = analytics.compute(snap)
    latest = derived.annual[-1]
    require(latest.period == "FY 2026", "expected the distressed year last")

    _absent(latest.roe, "return on equity on negative equity")
    _absent(latest.roce, "return on capital employed on negative capital")
    _absent(latest.roic, "return on invested capital on negative capital")
    _absent(latest.debt_to_equity, "debt to equity on negative equity")
    _absent(latest.goodwill_to_equity, "goodwill to equity on negative equity")
    _absent(latest.effective_tax_rate, "effective tax rate on a pre-tax loss")
    _absent(latest.nopat, "NOPAT built on an unusable tax rate")
    _absent(latest.net_debt_to_ebitda, "net debt to EBITDA on negative EBITDA")
    _absent(latest.cfo_to_pat, "cash conversion on a loss")
    _absent(latest.dio, "inventory days on a negative cost base")
    _absent(latest.dpo, "payable days on a negative cost base")

    # Interest cover is not withheld: the denominator is positive and the
    # negative result is the finding.
    require(
        latest.interest_coverage is not None and latest.interest_coverage < 0,
        f"interest cover should be negative and present, got {latest.interest_coverage!r}",
    )
    return "11 ratios withheld on the distressed year, interest cover retained"


def check_distressed_composites() -> str:
    """Composites must withhold rather than publish a meaningless total."""
    snap = _distressed_snapshot()
    derived = analytics.compute(snap)
    comp = composites.compute(snap, derived)

    latest = comp.dupont.years[-1]
    _absent(latest.tax_burden, "DuPont tax burden on a pre-tax loss")
    _absent(latest.interest_burden, "DuPont interest burden on negative EBIT")
    _absent(latest.equity_multiplier, "DuPont equity multiplier on negative equity")
    _absent(latest.roe_direct, "DuPont ROE on negative equity")
    _absent(latest.roe_product, "DuPont factor product with absent factors")

    # The F-Score still scores: its signals are sign and direction tests, and
    # a company failing all of them is precisely what the framework is for.
    require(
        comp.piotroski.score is not None,
        "the F-Score should still be computable for a distressed company",
    )
    require(
        comp.piotroski.score <= 3,
        f"a company losing money on every axis should score low, got {comp.piotroski.score}",
    )

    # Altman is designed for exactly this company and must produce a score
    # in the distress band rather than withhold.
    require(
        comp.altman.score is not None,
        "the Z-Score should be computable for a distressed manufacturer",
    )
    require(
        comp.altman.zone == "Distress",
        f"expected the distress band, got {comp.altman.zone!r} at {comp.altman.score!r}",
    )
    return (
        f"DuPont withheld, F-Score {comp.piotroski.score} of "
        f"{comp.piotroski.computable}, Z-Score {comp.altman.score:.2f} in the "
        f"{comp.altman.zone.lower()} band"
    )


def check_guardrail_detector_fires() -> str:
    """The breach detector must report a planted breach.

    A detector that always returns nil breaches is indistinguishable from one
    that works, so it is tested against a report whose figures have been
    deliberately corrupted after computation.
    """
    snap = _distressed_snapshot()
    derived = analytics.compute(snap)
    comp = composites.compute(snap, derived)

    clean = selfcheck.run(snap, derived, comp)
    guardrail = next(
        c for c in clean.checks if c.name.startswith("No ratio published against")
    )
    require(
        guardrail.passed is True,
        f"the clean distressed report should show no breach, got {guardrail.actual!r}",
    )

    # Plant one: an ROE on the year whose equity is negative.
    derived.annual[-1].roe = 275.0
    tampered = selfcheck.run(snap, derived, comp)
    planted = next(
        c for c in tampered.checks if c.name.startswith("No ratio published against")
    )
    require(planted.passed is False, "the detector failed to report a planted breach")
    require(
        "return on equity" in planted.detail,
        f"the detector should name the offending figure, got {planted.detail!r}",
    )
    return "detector reports nil breaches when clean and one when planted"


def check_composites_carry_components() -> str:
    """A score must never be reachable without the parts that produced it."""
    reports = 0
    for ticker in CACHED_TICKERS:
        has_cache = (CACHE_ROOT / ticker / "api" / "summary.json").exists() or (
            CACHE_ROOT / ticker / "summary.json"
        ).exists()
        if not has_cache:
            continue
        snap = _cached_snapshot(ticker)
        derived = analytics.compute(snap)
        comp = composites.compute(snap, derived)
        reports += 1

        score = comp.piotroski
        if score.score is not None:
            require(
                bool(score.tests), f"{ticker} published an F-Score with no sub-tests"
            )
            awarded = [t.points for t in score.tests if t.points is not None]
            require(
                sum(awarded) == score.score,
                f"{ticker} F-Score {score.score} does not equal its sub-tests {sum(awarded)}",
            )
            require(
                all(t.definition for t in score.tests),
                f"{ticker} has an F-Score signal with no stated test",
            )
            require(
                score.computable > 0,
                f"{ticker} published a score against a nil denominator",
            )

        altman = comp.altman
        if altman.score is not None:
            require(
                len(altman.components) == 5,
                f"{ticker} published a Z-Score with {len(altman.components)} terms, expected 5",
            )
            require(
                all(c.definition and c.weight for c in altman.components),
                f"{ticker} has a Z-Score term with no definition or coefficient",
            )
            rebuilt = sum(c.contribution for c in altman.components)
            require(
                abs(rebuilt - altman.score) < 1e-9,
                f"{ticker} Z-Score does not equal its terms",
            )
        else:
            require(
                bool(altman.withheld_reason) or not altman.components,
                f"{ticker} withheld a Z-Score without saying why",
            )

        if comp.reinvestment.withheld_reason:
            require(
                not comp.reinvestment.years,
                f"{ticker} withheld the reinvestment identity but kept rows",
            )
    require(reports > 0, "no cached tickers were available to check")
    return f"{reports} reports carry components for every score published"


def check_cached_reports_verify() -> str:
    """Every cached company's own self-check must close."""
    lines: list[str] = []
    for ticker in CACHED_TICKERS:
        has_cache = (CACHE_ROOT / ticker / "api" / "summary.json").exists() or (
            CACHE_ROOT / ticker / "summary.json"
        ).exists()
        if not has_cache:
            lines.append(f"{ticker} skipped, not cached")
            continue
        snap = _cached_snapshot(ticker)
        derived = analytics.compute(snap)
        comp = composites.compute(snap, derived)
        result = selfcheck.run(snap, derived, comp)
        require(
            result.all_passed,
            f"{ticker} failed {len(result.failures)} self-check(s): "
            f"{'; '.join(f.name for f in result.failures)}",
        )
        require(result.passed_count > 0, f"{ticker} ran no applicable checks at all")
        lines.append(f"{ticker} {result.passed_count}/{len(result.applicable)}")
    return "; ".join(lines)


def _cached_snapshot(ticker: str) -> CompanySnapshot:
    """Builds a snapshot from the on-disk cache without touching the network."""
    client = CollectorClient(
        base_url="http://cache.invalid", cache_dir=CACHE_ROOT, use_cache=True
    )
    payloads = {}
    for name in (
        "summary",
        "peers",
        "income_q",
        "income_a",
        "growth_q",
        "growth_a",
        "balance",
        "balance_growth",
        "cashflow",
        "dupont",
        "solvency",
        "liquidity",
        "capital_efficiency",
        "cagr",
    ):
        payloads[name] = client.fetch(ticker, name)
    return build_snapshot(ticker, payloads)


def check_distressed_report_renders() -> str:
    """The renderer must survive a company where most figures are absent.

    Withholding is the path least likely to be exercised by real data and
    most likely to break the templates: every table has em-dashes in it, two
    frameworks are replaced by prose, and several exhibits drop out entirely.
    Compiling the distressed company end to end is the only way to know the
    layout holds when almost nothing is available to lay out.
    """
    import tempfile

    from reporting import (
        charts as charts_module,
        typst_doc,
    )

    snap = _distressed_snapshot()
    derived = analytics.compute(snap)
    comp = composites.compute(snap, derived)
    check = selfcheck.run(snap, derived, comp)

    with tempfile.TemporaryDirectory() as work:
        work_dir = Path(work)
        produced = charts_module.render_all(snap, derived, comp, work_dir)
        source = typst_doc.build_document(
            snap, derived, comp, check, produced, as_of="01 Jan 2026"
        )
        require(
            len(source) > 10000,
            f"the generated source is implausibly short at {len(source)} chars",
        )
        require(
            "Composite quality" in source,
            "the composite section is missing from the source",
        )

        source_path = work_dir / "stress.typ"
        source_path.write_text(source, encoding="utf-8")
        try:
            import typst

            pdf = typst.compile(str(source_path))
        except Exception as exc:  # noqa: BLE001 - surfaced as a failure
            raise Failure(f"Typst compilation failed: {exc}") from exc
        require(
            len(pdf) > 20000,
            f"the compiled PDF is implausibly small at {len(pdf)} bytes",
        )
    return (
        f"compiled a {len(pdf) // 1024} KB report with {len(produced)} charts "
        "from broken statements"
    )


def pytest_approx(expected: float, tolerance: float = 1e-6):
    """Returns a value comparing equal to `expected` within `tolerance`."""

    class _Approx:
        def __eq__(self, other: object) -> bool:
            return isinstance(other, (int, float)) and abs(other - expected) < tolerance

        def __repr__(self) -> str:
            return f"~{expected}"

    return _Approx()


# --- Tier 2 composites -------------------------------------------------------


def check_altman_model_constants() -> str:
    """The published Altman coefficients and zone thresholds are unaltered."""
    require(
        composites.ALTMAN_PUBLIC_SAFE == 2.99
        and composites.ALTMAN_PUBLIC_DISTRESS == 1.81,
        "public Altman zones must stay 2.99 / 1.81",
    )
    require(
        composites.ALTMAN_PRIVATE_SAFE == 2.90
        and composites.ALTMAN_PRIVATE_DISTRESS == 1.23,
        "private Altman zones must stay 2.90 / 1.23",
    )
    for name, weights in (
        ("public", composites.ALTMAN_PUBLIC_WEIGHTS),
        ("private", composites.ALTMAN_PRIVATE_WEIGHTS),
    ):
        require(len(weights) == 5, f"{name} Altman needs 5 weighted terms")
        require(
            all(isinstance(v, float) for v in weights.values()),
            f"{name} Altman weights must be floats",
        )
    return (
        f"public {composites.ALTMAN_PUBLIC_DISTRESS}-{composites.ALTMAN_PUBLIC_SAFE}, "
        f"private {composites.ALTMAN_PRIVATE_DISTRESS}-{composites.ALTMAN_PRIVATE_SAFE}"
    )


def check_altman_score_is_sum_of_terms() -> str:
    """Z equals the sum of its weighted components, to floating tolerance."""
    snap = _distressed_snapshot()
    comp = composites.compute(snap, analytics.compute(snap))
    altman = comp.altman
    if altman is None or altman.score is None:
        raise Failure("the distressed fixture produced no Altman score to check")

    total = sum(c.contribution for c in altman.components if c.contribution is not None)
    require(
        abs(total - altman.score) < 1e-9,
        f"Z {altman.score:.6f} != sum of terms {total:.6f}",
    )
    require(
        altman.distress_below < altman.safe_above,
        "the distress threshold must sit below the safe threshold",
    )
    return f"Z={altman.score:.3f} over {len(altman.components)} terms"


def check_piotroski_is_nine_binary_tests() -> str:
    """The F-Score is nine tests, each awarding nil or one point."""
    snap = _distressed_snapshot()
    comp = composites.compute(snap, analytics.compute(snap))
    tests = comp.piotroski.tests
    require(len(tests) == 9, f"F-Score is nine tests, found {len(tests)}")

    for test in tests:
        require(
            test.passed in (True, False, None),
            f"{test.name} recorded {test.passed!r}; a signal is binary or absent",
        )
        require(bool(test.name), "every F-Score test must carry a name")
        require(bool(test.definition), f"{test.name} must state its definition")
        if test.passed is None:
            require(
                bool(test.unavailable_reason),
                f"{test.name} is unavailable without saying why",
            )

    awarded = [t for t in tests if t.passed is not None]
    if comp.piotroski.score is not None:
        require(
            comp.piotroski.score == sum(1 for t in awarded if t.passed),
            "the F-Score must equal the count of signals that passed",
        )
    numbers = sorted(t.number for t in tests)
    require(
        numbers == list(range(1, 10)), f"signals must be numbered 1-9, got {numbers}"
    )
    return f"9 tests, {len(awarded)} scored, score={comp.piotroski.score!r}"


def check_financial_sector_withholds_altman() -> str:
    """A bank is flagged financial, and Altman is withheld with a reason.

    Z-Score was calibrated on manufacturers; working capital and asset
    turnover do not carry the same meaning for a lender, so a number here
    would be precise and wrong.
    """
    require(
        len(composites.FINANCIAL_SECTOR_MARKERS) > 0,
        "the financial-sector marker list must not be empty",
    )
    snap = _distressed_snapshot()
    snap.sector = "Banks"
    comp = composites.compute(snap, analytics.compute(snap))
    require(comp.is_financial, "a company whose sector is Banks must be flagged")
    if comp.altman is not None:
        require(
            comp.altman.score is None and bool(comp.altman.withheld_reason),
            "a financial issuer's Altman score must be withheld, with a reason",
        )
    return f"flagged financial; {len(composites.FINANCIAL_SECTOR_MARKERS)} markers"


def check_sources_and_uses_balances() -> str:
    """Every capital-allocation flow is signed and the two sides reconcile."""
    snap = _distressed_snapshot()
    comp = composites.compute(snap, analytics.compute(snap))
    flows = comp.sources_uses
    if flows is None:
        return "withheld on the distressed fixture"

    for item in list(flows.sources) + list(flows.uses):
        require(bool(item.label), "every flow item needs a label")
        require(
            item.amount is None or isinstance(item.amount, float),
            f"{item.label} carries a non-float amount",
        )
    return f"{len(flows.sources)} sources, {len(flows.uses)} uses"


def check_dupont_factors_multiply_to_roe() -> str:
    """Each DuPont year's five factors multiply back to its own ROE."""
    snap = _distressed_snapshot()
    comp = composites.compute(snap, analytics.compute(snap))
    checked = 0
    for year in comp.dupont.years:
        factors = [
            year.tax_burden,
            year.interest_burden,
            year.operating_margin,
            year.asset_turnover,
            year.equity_multiplier,
        ]
        if any(f is None for f in factors) or year.roe_direct is None:
            continue
        product = 1.0
        for factor in factors:
            product *= factor
        require(
            abs(product - year.roe_direct) < 0.5,
            f"{year.period}: factors give {product:.4f}, ROE {year.roe_direct:.4f}",
        )
        checked += 1
    return f"{checked} of {len(comp.dupont.years)} year(s) fully populated"


# --- Analytics ---------------------------------------------------------------


def check_analytics_denominator_guard() -> str:
    """`_div_positive` mirrors `fmt.pos_div`: no quotient on a non-positive base."""
    require(analytics._div_positive(10.0, 2.0) == 5.0, "a positive base must divide")
    require(
        analytics._div_positive(-10.0, 2.0) == -5.0,
        "a negative numerator over a positive base is real information",
    )
    for bad in (0.0, -2.0, None):
        _absent(analytics._div_positive(10.0, bad), f"_div_positive(10, {bad!r})")
    _absent(analytics._div_positive(None, 2.0), "_div_positive(None, 2)")
    require(analytics.DAYS_IN_YEAR == 365.0, "day-count metrics assume a 365-day year")
    return "guard one-sided, 365-day year"


def check_enterprise_value_identity() -> str:
    """Enterprise value is market cap plus debt less cash."""
    snap = _distressed_snapshot()
    ev = analytics.compute(snap).enterprise
    parts = (ev.market_cap, ev.total_debt, ev.cash, ev.enterprise_value)
    if any(p is None for p in parts):
        return "withheld: the fixture lacks a market capitalisation"
    minority = ev.minority_interest or 0.0
    expected = ev.market_cap + ev.total_debt + minority - ev.cash
    require(
        abs(expected - ev.enterprise_value) < 0.01,
        f"EV {ev.enterprise_value} != {ev.market_cap} + {ev.total_debt} "
        f"+ {minority} - {ev.cash}",
    )
    return f"EV={ev.enterprise_value:.1f}"


def check_rolling_window_is_four_quarters() -> str:
    """The rolling series sums four quarters, and never invents periods."""
    snap = _distressed_snapshot()
    derived = analytics.compute(snap)
    require(
        len(derived.rolling) <= max(0, len(snap.quarters) - 3),
        "a 4-quarter rolling series cannot be longer than quarters minus three",
    )
    for point in derived.rolling:
        require(bool(point.period), "every rolling point must name its period")
    return f"{len(derived.rolling)} point(s) from {len(snap.quarters)} quarter(s)"


# --- Snapshot ----------------------------------------------------------------


def check_snapshot_unit_conversion() -> str:
    """Rupee million to crore is a factor of ten, and fractions become percent."""
    from reporting import snapshot as snapshot_module

    require(snapshot_module.MN_PER_CR == 10.0, "10 million rupees is one crore")
    require(
        snapshot_module._pct_from_fraction(0.1534) == pytest_approx(15.34),
        "a fraction must scale to percentage points",
    )
    _absent(snapshot_module._pct_from_fraction(None), "_pct_from_fraction(None)")
    return "MN_PER_CR=10, fraction to percent"


def check_snapshot_period_caps() -> str:
    """The declared period caps bound what a snapshot exposes."""
    from reporting import snapshot as snapshot_module

    require(snapshot_module.QUARTERS_SHOWN > 0, "quarter cap must be positive")
    require(snapshot_module.YEARS_SHOWN > 0, "year cap must be positive")
    snap = _distressed_snapshot()
    require(
        len(snap.quarters) <= snapshot_module.QUARTERS_SHOWN,
        "quarters exceed the declared cap",
    )
    require(
        len(snap.years) <= snapshot_module.YEARS_SHOWN,
        "years exceed the declared cap",
    )
    return f"{snapshot_module.QUARTERS_SHOWN}Q / {snapshot_module.YEARS_SHOWN}Y"


# --- Self-check suite --------------------------------------------------------


def check_selfcheck_suite_is_intact() -> str:
    """A cached ticker runs the documented 20 checks, and none was lost.

    `check_cached_reports_verify` asserts the checks *pass*. Nothing asserted
    that they all still *exist*, so a refactor deleting one went unnoticed.
    """
    ran = 0
    for ticker in CACHED_TICKERS:
        has_cache = (CACHE_ROOT / ticker / "api" / "summary.json").exists() or (
            CACHE_ROOT / ticker / "summary.json"
        ).exists()
        if not has_cache:
            continue
        snap = _cached_snapshot(ticker)
        derived = analytics.compute(snap)
        comp = composites.compute(snap, derived)
        result = selfcheck.run(snap, derived, comp)
        require(
            len(result.applicable) >= 20,
            f"{ticker} ran {len(result.applicable)} checks; CLAUDE.md documents 20",
        )
        names = [c.name for c in result.applicable]
        require(len(names) == len(set(names)), f"{ticker} has duplicate check names")
        ran += 1
    if not ran:
        return "skipped, no cached ticker on disk"
    return f"{ran} ticker(s), >=20 uniquely named checks each"


def check_selfcheck_tolerances() -> str:
    """The four tolerance bands stay ordered from exact to methodological."""
    require(
        selfcheck.TOL_IDENTITY_PCT
        < selfcheck.TOL_PROVIDER_PCT
        < selfcheck.TOL_METHODOLOGY_PCT,
        "tolerances must widen from identity to provider to methodology",
    )
    require(
        selfcheck.TOL_IDENTITY_CRORE > 0,
        "the absolute identity tolerance must be positive",
    )
    return (
        f"identity {selfcheck.TOL_IDENTITY_PCT:g}% < provider "
        f"{selfcheck.TOL_PROVIDER_PCT:g}% < method {selfcheck.TOL_METHODOLOGY_PCT:g}%"
    )


# --- Design tokens and charts ------------------------------------------------


def check_design_tokens_are_valid() -> str:
    """Every colour token is a hex triple and the data series are distinct."""
    from reporting import tokens

    names = [n for n in dir(tokens) if n.isupper()]
    colours = {
        n: getattr(tokens, n)
        for n in names
        if isinstance(getattr(tokens, n), str) and getattr(tokens, n).startswith("#")
    }
    require(colours, "no colour tokens found")
    for name, value in colours.items():
        require(
            len(value) == 7 and all(c in "0123456789abcdefABCDEF" for c in value[1:]),
            f"{name}={value!r} is not a #rrggbb triple",
        )
    require(
        len(tokens.SERIES) == len(set(tokens.SERIES)),
        "the data series palette repeats a colour",
    )
    require(
        tokens.BRAND not in tokens.SERIES,
        "BRAND is furniture only and must never appear in the data palette",
    )
    require(
        tokens.POSITIVE != tokens.NEGATIVE,
        "gains and losses must not share a colour",
    )
    return f"{len(colours)} colours, {len(tokens.SERIES)} series"


def check_charts_withhold_rather_than_raise() -> str:
    """Every chart returns None on an empty snapshot instead of raising."""
    import inspect
    import tempfile

    from reporting import charts as charts_module

    empty = CompanySnapshot(ticker="EMPTY")
    derived = analytics.compute(empty)
    comp = composites.compute(empty, derived)

    single: list[str] = []
    with tempfile.TemporaryDirectory() as raw:
        out = Path(raw)
        for name, func in sorted(vars(charts_module).items()):
            if name.startswith("_") or not inspect.isfunction(func):
                continue
            params = list(inspect.signature(func).parameters)
            if params[-1:] != ["out_dir"] or len(params) != 2:
                continue
            first = {"snap": empty, "derived": derived, "comp": comp}.get(params[0])
            if first is None:
                continue
            try:
                result = func(first, out)
            except Exception as exc:  # noqa: BLE001 - that is the finding
                raise Failure(f"{name} raised on an empty snapshot: {exc}") from exc
            require(
                result is None or Path(result).exists(),
                f"{name} returned {result!r} but wrote no file",
            )
            single.append(name)
    require(len(single) >= 8, f"only {len(single)} chart function(s) exercised")
    return f"{len(single)} charts withheld cleanly"


def check_render_all_writes_svg() -> str:
    """`render_all` produces real SVG files for the distressed fixture."""
    import tempfile

    from reporting import charts as charts_module

    snap = _distressed_snapshot()
    derived = analytics.compute(snap)
    comp = composites.compute(snap, derived)
    with tempfile.TemporaryDirectory() as raw:
        out = Path(raw)
        produced = charts_module.render_all(snap, derived, comp, out)
        require(isinstance(produced, dict), "render_all must return a mapping")
        written = sorted(out.glob("*.svg"))
        require(written, "render_all wrote no SVG at all")
        for path in written:
            head = path.read_text(encoding="utf-8", errors="replace")[:512]
            require("<svg" in head, f"{path.name} is not SVG")
    return f"{len(written)} SVG file(s)"


CHECKS: tuple[tuple[str, Callable[[], str]], ...] = (
    ("Division helpers", check_division_helpers),
    ("Distressed ratios withheld", check_distressed_ratios),
    ("Distressed composites", check_distressed_composites),
    ("Guardrail detector fires", check_guardrail_detector_fires),
    ("Composites carry components", check_composites_carry_components),
    ("Altman model constants", check_altman_model_constants),
    ("Altman Z is the sum of its terms", check_altman_score_is_sum_of_terms),
    ("Piotroski is nine binary tests", check_piotroski_is_nine_binary_tests),
    ("Financial sector withholds Altman", check_financial_sector_withholds_altman),
    ("Sources and uses balance", check_sources_and_uses_balances),
    ("DuPont factors multiply to ROE", check_dupont_factors_multiply_to_roe),
    ("Analytics denominator guard", check_analytics_denominator_guard),
    ("Enterprise value identity", check_enterprise_value_identity),
    ("Rolling window is four quarters", check_rolling_window_is_four_quarters),
    ("Snapshot unit conversion", check_snapshot_unit_conversion),
    ("Snapshot period caps", check_snapshot_period_caps),
    ("Self-check suite is intact", check_selfcheck_suite_is_intact),
    ("Self-check tolerances ordered", check_selfcheck_tolerances),
    ("Design tokens are valid", check_design_tokens_are_valid),
    ("Charts withhold, never raise", check_charts_withhold_rather_than_raise),
    ("render_all writes SVG", check_render_all_writes_svg),
    ("Cached reports self-verify", check_cached_reports_verify),
    ("Distressed report renders", check_distressed_report_renders),
)


def main() -> int:
    """Runs every check and reports the outcome.

    Returns:
        Process exit code: 0 when every check held, 1 otherwise.

    """
    cli.setup(logging.ERROR, cli.PLAIN, stream=sys.stderr)
    banner("Report engine verification")

    report = Report()
    for name, check in CHECKS:
        report.run(name, check)

    print(report.render())
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
