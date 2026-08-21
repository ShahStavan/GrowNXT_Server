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
from pathlib import Path
import sys
from typing import Callable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import cli  # noqa: E402
from scripts.checks import Failure, Report, banner, require  # noqa: E402
from reporting import analytics, composites, fmt, selfcheck  # noqa: E402
from reporting.client import CollectorClient  # noqa: E402
from reporting.snapshot import (  # noqa: E402
    BalancePeriod, CashflowPeriod, CompanySnapshot, IncomePeriod,
    build_snapshot,
)

# Tickers with a complete payload set already on disk. The harness reads the
# cache only; it must not depend on the network to run.
CACHED_TICKERS: Tuple[str, ...] = ("WIPRO", "RELIANCE", "TCS", "HDFCBANK")

CACHE_ROOT: Path = Path(__file__).resolve().parent.parent / ".cache" / "api"


def _absent(value, label: str) -> None:
    """Asserts a ratio was withheld rather than computed."""
    require(value is None,
             "%s should have been withheld but came back as %r" % (label, value))


def _distressed_snapshot() -> CompanySnapshot:
    """Builds a company that breaks every denominator at once.

    Equity turns negative, EBIT and EBITDA turn negative, pre-tax profit
    turns negative, the cost base is reported negative, and capital employed
    and invested capital both fall below zero. Each of those is individually
    plausible in a real filing; together they exercise every guard in one
    pass.
    """
    years = [
        IncomePeriod(period="FY 2024", revenue=4000.0, ebit=200.0, pbt=100.0,
                     pat=80.0, eps=0.8, depreciation=120.0, interest=100.0,
                     raw_materials=2400.0, sga=600.0),
        IncomePeriod(period="FY 2025", revenue=3500.0, ebit=-300.0, pbt=-500.0,
                     pat=-600.0, eps=-6.0, depreciation=130.0, interest=200.0,
                     raw_materials=2200.0, sga=650.0),
        # The distressed year. EBITDA is EBIT plus depreciation, so -800 plus
        # 100 leaves it negative too.
        IncomePeriod(period="FY 2026", revenue=3000.0, ebit=-800.0, pbt=-1200.0,
                     pat=-1100.0, eps=-11.0, depreciation=100.0, interest=300.0,
                     raw_materials=-50.0, sga=700.0),
    ]
    balance = [
        BalancePeriod(period="FY 2024", equity=1000.0, debt=2000.0, cash=300.0,
                      total_assets=5000.0, current_assets=1800.0,
                      current_liabilities=1500.0, inventory=400.0,
                      receivables=500.0, shares_cr=100.0, payables=300.0,
                      goodwill_intangibles=200.0, retained_earnings=400.0,
                      long_term_debt=1500.0, total_liabilities=4000.0),
        BalancePeriod(period="FY 2025", equity=200.0, debt=2600.0, cash=200.0,
                      total_assets=4800.0, current_assets=1600.0,
                      current_liabilities=2000.0, inventory=380.0,
                      receivables=480.0, shares_cr=100.0, payables=320.0,
                      goodwill_intangibles=200.0, retained_earnings=-200.0,
                      long_term_debt=1900.0, total_liabilities=4600.0),
        # Negative equity; current liabilities above total assets, so capital
        # employed is negative; and cash above equity plus debt, so invested
        # capital is negative.
        BalancePeriod(period="FY 2026", equity=-400.0, debt=2900.0, cash=3400.0,
                      total_assets=4500.0, current_assets=1400.0,
                      current_liabilities=4800.0, inventory=360.0,
                      receivables=450.0, shares_cr=100.0, payables=350.0,
                      goodwill_intangibles=200.0, retained_earnings=-900.0,
                      long_term_debt=2100.0, total_liabilities=4900.0),
    ]
    cashflow = [
        CashflowPeriod(period="FY 2024", cfo=300.0, capex=150.0, fcf=150.0,
                       dividends_paid=40.0, cash_from_investing=-150.0,
                       cash_from_financing=-100.0, net_change_in_cash=50.0),
        CashflowPeriod(period="FY 2025", cfo=-100.0, capex=120.0, fcf=-220.0,
                       dividends_paid=0.0, cash_from_investing=-120.0,
                       cash_from_financing=120.0, net_change_in_cash=-100.0),
        CashflowPeriod(period="FY 2026", cfo=-500.0, capex=90.0, fcf=-590.0,
                       dividends_paid=0.0, cash_from_investing=-90.0,
                       cash_from_financing=390.0, net_change_in_cash=-200.0),
    ]
    return CompanySnapshot(
        ticker="STRESS", name="Stress Test Industries",
        sector="Diversified Manufacturing",
        years=years, balance=balance, cashflow=cashflow,
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
    require(result is not None and abs(result + 2.0) < 1e-12,
             "a negative numerator over a positive base must survive, got %r"
             % result)
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
    require(latest.interest_coverage is not None and latest.interest_coverage < 0,
             "interest cover should be negative and present, got %r"
             % latest.interest_coverage)
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
    require(comp.piotroski.score is not None,
             "the F-Score should still be computable for a distressed company")
    require(comp.piotroski.score <= 3,
             "a company losing money on every axis should score low, got %s"
             % comp.piotroski.score)

    # Altman is designed for exactly this company and must produce a score
    # in the distress band rather than withhold.
    require(comp.altman.score is not None,
             "the Z-Score should be computable for a distressed manufacturer")
    require(comp.altman.zone == "Distress",
             "expected the distress band, got %r at %r"
             % (comp.altman.zone, comp.altman.score))
    return "DuPont withheld, F-Score %d of %d, Z-Score %.2f in the %s band" % (
        comp.piotroski.score, comp.piotroski.computable,
        comp.altman.score, comp.altman.zone.lower())


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
    guardrail = next(c for c in clean.checks
                     if c.name.startswith("No ratio published against"))
    require(guardrail.passed is True,
             "the clean distressed report should show no breach, got %r"
             % guardrail.actual)

    # Plant one: an ROE on the year whose equity is negative.
    derived.annual[-1].roe = 275.0
    tampered = selfcheck.run(snap, derived, comp)
    planted = next(c for c in tampered.checks
                   if c.name.startswith("No ratio published against"))
    require(planted.passed is False,
             "the detector failed to report a planted breach")
    require("return on equity" in planted.detail,
             "the detector should name the offending figure, got %r"
             % planted.detail)
    return "detector reports nil breaches when clean and one when planted"


def check_composites_carry_components() -> str:
    """A score must never be reachable without the parts that produced it."""
    reports = 0
    for ticker in CACHED_TICKERS:
        if not (CACHE_ROOT / ticker / "summary.json").exists():
            continue
        snap = _cached_snapshot(ticker)
        derived = analytics.compute(snap)
        comp = composites.compute(snap, derived)
        reports += 1

        score = comp.piotroski
        if score.score is not None:
            require(bool(score.tests),
                     "%s published an F-Score with no sub-tests" % ticker)
            awarded = [t.points for t in score.tests if t.points is not None]
            require(sum(awarded) == score.score,
                     "%s F-Score %s does not equal its sub-tests %s"
                     % (ticker, score.score, sum(awarded)))
            require(all(t.definition for t in score.tests),
                     "%s has an F-Score signal with no stated test" % ticker)
            require(score.computable > 0,
                     "%s published a score against a nil denominator" % ticker)

        altman = comp.altman
        if altman.score is not None:
            require(len(altman.components) == 5,
                     "%s published a Z-Score with %d terms, expected 5"
                     % (ticker, len(altman.components)))
            require(all(c.definition and c.weight for c in altman.components),
                     "%s has a Z-Score term with no definition or coefficient"
                     % ticker)
            rebuilt = sum(c.contribution for c in altman.components)
            require(abs(rebuilt - altman.score) < 1e-9,
                     "%s Z-Score does not equal its terms" % ticker)
        else:
            require(bool(altman.withheld_reason) or not altman.components,
                     "%s withheld a Z-Score without saying why" % ticker)

        if comp.reinvestment.withheld_reason:
            require(not comp.reinvestment.years,
                     "%s withheld the reinvestment identity but kept rows"
                     % ticker)
    require(reports > 0, "no cached tickers were available to check")
    return "%d reports carry components for every score published" % reports


def check_cached_reports_verify() -> str:
    """Every cached company's own self-check must close."""
    lines: List[str] = []
    for ticker in CACHED_TICKERS:
        if not (CACHE_ROOT / ticker / "summary.json").exists():
            lines.append("%s skipped, not cached" % ticker)
            continue
        snap = _cached_snapshot(ticker)
        derived = analytics.compute(snap)
        comp = composites.compute(snap, derived)
        result = selfcheck.run(snap, derived, comp)
        require(result.all_passed,
                 "%s failed %d self-check(s): %s"
                 % (ticker, len(result.failures),
                    "; ".join(f.name for f in result.failures)))
        require(result.passed_count > 0,
                 "%s ran no applicable checks at all" % ticker)
        lines.append("%s %d/%d" % (ticker, result.passed_count,
                                   len(result.applicable)))
    return "; ".join(lines)


def _cached_snapshot(ticker: str) -> CompanySnapshot:
    """Builds a snapshot from the on-disk cache without touching the network."""
    client = CollectorClient(base_url="http://cache.invalid", use_cache=True)
    payloads = {}
    for name in ("summary", "peers", "income_q", "income_a", "growth_q",
                 "growth_a", "balance", "balance_growth", "cashflow", "dupont",
                 "solvency", "liquidity", "capital_efficiency", "cagr"):
        path = CACHE_ROOT / ticker / (name + ".json")
        payloads[name] = client.fetch(ticker, name) if path.exists() else None
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

    from reporting import charts as charts_module
    from reporting import typst_doc

    snap = _distressed_snapshot()
    derived = analytics.compute(snap)
    comp = composites.compute(snap, derived)
    check = selfcheck.run(snap, derived, comp)

    with tempfile.TemporaryDirectory() as work:
        work_dir = Path(work)
        produced = charts_module.render_all(snap, derived, comp, work_dir)
        source = typst_doc.build_document(
            snap, derived, comp, check, produced, as_of="01 Jan 2026")
        require(len(source) > 10000,
                 "the generated source is implausibly short at %d chars"
                 % len(source))
        require("Composite quality" in source,
                 "the composite section is missing from the source")

        source_path = work_dir / "stress.typ"
        source_path.write_text(source, encoding="utf-8")
        try:
            import typst
            pdf = typst.compile(str(source_path))
        except Exception as exc:  # noqa: BLE001 - surfaced as a failure
            raise Failure("Typst compilation failed: %s" % exc)
        require(len(pdf) > 20000,
                 "the compiled PDF is implausibly small at %d bytes" % len(pdf))
    return "compiled a %d KB report with %d charts from broken statements" % (
        len(pdf) // 1024, len(produced))


CHECKS: Tuple[Tuple[str, Callable[[], str]], ...] = (
    ("Division helpers", check_division_helpers),
    ("Distressed ratios withheld", check_distressed_ratios),
    ("Distressed composites", check_distressed_composites),
    ("Guardrail detector fires", check_guardrail_detector_fires),
    ("Composites carry components", check_composites_carry_components),
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
