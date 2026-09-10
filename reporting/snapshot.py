"""Normalised domain model assembled from collector payloads.

The report templates read only this model, never raw JSON. That boundary is
where the unit contract is enforced and where the collector's quirks are
absorbed once instead of in every table:

    - `income/annual` appends a 'TTM' row that is not a fiscal year. It is
      split out rather than plotted alongside FY columns.
    - Series arrive oldest-first and of varying length (15 quarters, 11
      annual rows), so windows are taken from the tail.
    - `balTcso` is share count in crore, cross-checked against PAT / EPS.
    - Peer `marketCap` is denominated in rupee MILLION while every other
      figure is rupee CRORE. Converted on ingest; see MN_PER_CR.
    - The `_comments` growth annotations are unusable, so growth is derived.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from reporting import fmt

logger = logging.getLogger(__name__)

# One crore equals ten million, used to normalise peer market caps.
MN_PER_CR: float = 10.0

# Presentation decides how many periods a table shows (see typst_doc);
# normalisation keeps everything the provider reported, because derived
# metrics need a prior period for deltas and two-point averages.
QUARTERS_SHOWN: int = 12
YEARS_SHOWN: int = 11


def _f(value: Any) -> float | None:
    """Coerces a payload value to float, mapping absent/invalid to None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if result == result else None
    return None


def _pct_from_fraction(value: float | None) -> float | None:
    """Scales a 0-1 fraction into percentage points."""
    return None if value is None else value * 100.0


def _rows(payload: Any) -> list[dict[str, Any]]:
    """Extracts a list of period dictionaries from a payload."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in (
            "quarterlyData",
            "annualData",
            "balancesheetData",
            "cashflowData",
            "data",
        ):
            inner = payload.get(key)
            if isinstance(inner, list):
                return [r for r in inner if isinstance(r, dict)]
    return []


@dataclass
class IncomePeriod:
    """One income-statement period, quarterly or annual."""

    period: str
    revenue: float | None = None
    ebit: float | None = None
    pat: float | None = None
    eps: float | None = None
    depreciation: float | None = None
    interest: float | None = None
    pbt: float | None = None
    raw_materials: float | None = None
    sga: float | None = None
    dps: float | None = None
    payout_pct: float | None = None

    @property
    def raw_material_ratio(self) -> float | None:
        """Raw-material cost as a percentage of revenue.

        Note this is NOT cost of goods sold. The provider's `incRaw` field
        carries raw materials only: verified at 64.3% of revenue for
        Reliance but 0.6% for Wipro and 0.0% for TCS, whose costs are
        overwhelmingly people rather than materials. Subtracting it from
        revenue therefore does not yield gross profit, and a "gross margin"
        built that way reads as ~99% for any services business. Gross
        margin is deliberately absent from this model for that reason;
        EBITDA margin is the comparable profitability measure across
        sectors. `incGpro` would have been the correct source but is null
        for every ticker tested.
        """
        return fmt.margin(self.raw_materials, self.revenue)

    @property
    def sga_ratio(self) -> float | None:
        """Selling, general and administrative cost over revenue."""
        return fmt.margin(self.sga, self.revenue)

    @property
    def depreciation_ratio(self) -> float | None:
        """Depreciation and amortisation over revenue."""
        return fmt.margin(self.depreciation, self.revenue)

    @property
    def ebitda_margin(self) -> float | None:
        """EBITDA as a percentage of revenue."""
        return fmt.margin(self.ebitda, self.revenue)

    @property
    def effective_tax_rate(self) -> float | None:
        """Tax as a percentage of pre-tax profit.

        Withheld against a pre-tax loss. A loss carrying a tax credit gives
        an arithmetically valid rate that means nothing, and it would
        propagate: NOPAT is EBIT times one-less-this-rate, so a bad rate
        silently corrupts return on invested capital downstream.
        """
        retained = fmt.pos_div(self.pat, self.pbt)
        return None if retained is None else (1.0 - retained) * 100.0

    @property
    def ebitda(self) -> float | None:
        """EBIT plus depreciation, when both are present."""
        if self.ebit is None or self.depreciation is None:
            return None
        return self.ebit + self.depreciation

    @property
    def ebit_margin(self) -> float | None:
        """EBIT as a percentage of revenue."""
        return fmt.margin(self.ebit, self.revenue)

    @property
    def pat_margin(self) -> float | None:
        """PAT as a percentage of revenue."""
        return fmt.margin(self.pat, self.revenue)


@dataclass
class BalancePeriod:
    """One balance-sheet period."""

    period: str
    equity: float | None = None
    debt: float | None = None
    cash: float | None = None
    total_assets: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None
    inventory: float | None = None
    receivables: float | None = None
    net_ppe: float | None = None
    shares_cr: float | None = None
    payables: float | None = None
    goodwill_intangibles: float | None = None
    retained_earnings: float | None = None
    long_term_debt: float | None = None
    total_liabilities: float | None = None
    minority_interest: float | None = None

    @property
    def working_capital(self) -> float | None:
        """Current assets less current liabilities."""
        if self.current_assets is None or self.current_liabilities is None:
            return None
        return self.current_assets - self.current_liabilities

    @property
    def short_term_debt(self) -> float | None:
        """Total debt less the long-term portion."""
        if self.debt is None or self.long_term_debt is None:
            return None
        return max(0.0, self.debt - self.long_term_debt)

    @property
    def net_debt(self) -> float | None:
        """Total debt less cash and short-term investments."""
        if self.debt is None or self.cash is None:
            return None
        return self.debt - self.cash

    @property
    def debt_to_equity(self) -> float | None:
        """Total debt over shareholders' equity.

        Withheld against negative equity, where the ratio inverts its sign
        and reads as deleveraging at the exact moment the balance sheet is
        most stretched.
        """
        return fmt.pos_div(self.debt, self.equity)

    @property
    def book_value_per_share(self) -> float | None:
        """Equity per share, in rupees.

        Equity is in rupee crore and share count in crore, so the quotient
        is already rupees per share.
        """
        return fmt.pos_div(self.equity, self.shares_cr)


@dataclass
class CashflowPeriod:
    """One cash-flow period."""

    period: str
    cfo: float | None = None
    capex: float | None = None
    fcf: float | None = None
    dividends_paid: float | None = None
    change_in_working_capital: float | None = None
    cash_from_financing: float | None = None
    cash_from_investing: float | None = None
    # The provider's own closing figure for the year's movement in cash.
    # Held so that `selfcheck` can test the statement's articulation -
    # operating plus investing plus financing must equal it - rather than
    # assuming it. Capital allocation is built on that identity.
    net_change_in_cash: float | None = None


@dataclass
class Peer:
    """A listed peer with its valuation ratios."""

    ticker: str
    name: str
    market_cap_cr: float | None = None
    pe: float | None = None
    pb: float | None = None
    div_yield: float | None = None
    change_52w: float | None = None
    buy_reco_pct: float | None = None
    is_subject: bool = False


@dataclass
class Holding:
    """Shareholding pattern at one quarter end."""

    date: str
    promoter: float | None = None
    fii: float | None = None
    dii: float | None = None
    mutual_fund: float | None = None
    insider: float | None = None
    retail_other: float | None = None


@dataclass
class Dividend:
    """One declared dividend."""

    ex_date: str
    amount: float | None = None
    kind: str = ""


@dataclass
class CompanySnapshot:
    """Everything one report needs, normalised and unit-consistent."""

    ticker: str
    name: str = ""
    sector: str = ""
    description: str = ""

    quarters: list[IncomePeriod] = field(default_factory=list)
    years: list[IncomePeriod] = field(default_factory=list)
    ttm: IncomePeriod | None = None
    balance: list[BalancePeriod] = field(default_factory=list)
    cashflow: list[CashflowPeriod] = field(default_factory=list)

    peers: list[Peer] = field(default_factory=list)
    holdings: list[Holding] = field(default_factory=list)
    dividends: list[Dividend] = field(default_factory=list)

    key_ratios: dict[str, float] = field(default_factory=dict)
    dupont: dict[str, Any] = field(default_factory=dict)
    solvency: dict[str, Any] = field(default_factory=dict)
    liquidity: dict[str, Any] = field(default_factory=dict)
    capital_efficiency: dict[str, Any] = field(default_factory=dict)
    cagr: dict[str, Any] = field(default_factory=dict)

    total_reco: int | None = None
    buy_reco_pct: float | None = None
    market_cap_cr: float | None = None
    warnings: list[str] = field(default_factory=list)

    # --- derived -------------------------------------------------------

    @property
    def latest_balance(self) -> BalancePeriod | None:
        """Most recent balance-sheet period."""
        return self.balance[-1] if self.balance else None

    @property
    def shares_cr(self) -> float | None:
        """Share count in crore, from the latest balance sheet."""
        latest = self.latest_balance
        return latest.shares_cr if latest else None

    @property
    def ttm_eps(self) -> float | None:
        """Trailing twelve-month earnings per share."""
        if self.ttm is not None and self.ttm.eps is not None:
            return self.ttm.eps
        return self.years[-1].eps if self.years else None

    @property
    def implied_price(self) -> float | None:
        """Share price implied by the TTM P/E and TTM EPS.

        The collector exposes no live quote, so price is inferred. It is a
        derivation, not a market print, and is labelled as such wherever it
        appears.
        """
        pe = self.key_ratios.get("ttmPe")
        eps = self.ttm_eps
        if pe is None or eps is None:
            return None
        return pe * eps

    @property
    def implied_market_cap_cr(self) -> float | None:
        """Market capitalisation implied by price times share count."""
        price = self.implied_price
        shares = self.shares_cr
        if price is None or shares is None:
            return None
        return price * shares

    @property
    def pe_discount_to_industry(self) -> float | None:
        """Percentage discount or premium of P/E against the industry."""
        pe = self.key_ratios.get("ttmPe")
        industry = self.key_ratios.get("indpe")
        relative = fmt.pos_div(pe, industry)
        return None if relative is None else (relative - 1.0) * 100.0


def _income_rows(payload: Any, prefix: str) -> list[IncomePeriod]:
    """Maps income payload rows to IncomePeriod objects.

    Args:
        payload: Raw endpoint payload.
        prefix: Field prefix, 'qInc' for quarterly or 'inc' for annual.

    Returns:
        Periods in payload order (oldest first).
    """
    out: list[IncomePeriod] = []
    for row in _rows(payload):
        out.append(
            IncomePeriod(
                period=str(row.get("displayPeriod") or ""),
                revenue=_f(row.get(prefix + "Trev")),
                ebit=_f(row.get(prefix + "Ebi")),
                pat=_f(row.get(prefix + "Ninc")),
                eps=_f(row.get(prefix + "Eps")),
                depreciation=_f(row.get(prefix + "Dep")),
                interest=_f(row.get(prefix + "Ioi")),
                pbt=_f(row.get(prefix + "Pbt")),
                raw_materials=_f(row.get(prefix + "Raw")),
                sga=_f(row.get(prefix + "Sga")),
                dps=_f(row.get(prefix + "Dps")),
                # `incPyr` arrives as a fraction, not percentage points:
                # 0.8735 where DPS/EPS is 0.874. Scaled here so every ratio in
                # the model is in the same units.
                payout_pct=_pct_from_fraction(_f(row.get(prefix + "Pyr"))),
            )
        )
    return out


def _ratio_block(payload: Any, key: str) -> dict[str, Any]:
    """Extracts a named ratio sub-dictionary plus its period label."""
    if not isinstance(payload, dict):
        return {}
    block = payload.get(key)
    result: dict[str, Any] = dict(block) if isinstance(block, dict) else {}
    if "period" in payload:
        result["period"] = payload["period"]
    raw = payload.get("raw_variables")
    if isinstance(raw, dict):
        result["raw"] = raw
    return result


def _apply_income(snap: CompanySnapshot, payloads: dict[str, Any]) -> None:
    """Fills the quarterly and annual income statements.

    The annual endpoint returns the trailing twelve months as one more row
    alongside the fiscal years. It is separated here rather than downstream,
    because a TTM row left in a fiscal-year series silently becomes an extra
    year in every growth rate and every average computed from it.
    """
    quarters = _income_rows(payloads.get("income_q"), "qInc")
    snap.quarters = quarters[-QUARTERS_SHOWN:] if quarters else []
    if not quarters:
        snap.warnings.append("quarterly income statement unavailable")

    annual = _income_rows(payloads.get("income_a"), "inc")
    trailing = [r for r in annual if r.period.strip().upper() == "TTM"]
    fiscal = [r for r in annual if r.period.strip().upper() != "TTM"]
    snap.ttm = trailing[-1] if trailing else None
    snap.years = fiscal[-YEARS_SHOWN:] if fiscal else []
    if not fiscal:
        snap.warnings.append("annual income statement unavailable")


def _balance_periods(payload: Any) -> list[BalancePeriod]:
    """Parses the balance sheet endpoint into the periods shown."""
    return [
        BalancePeriod(
            period=str(r.get("displayPeriod") or ""),
            equity=_f(r.get("balTeq")),
            debt=_f(r.get("balTdeb")),
            cash=_f(r.get("balCsti")),
            total_assets=_f(r.get("balTota")),
            current_assets=_f(r.get("balTca")),
            current_liabilities=_f(r.get("balTcl")),
            inventory=_f(r.get("balTinv")),
            receivables=_f(r.get("balTrec")),
            net_ppe=_f(r.get("balNppe")),
            shares_cr=_f(r.get("balTcso")),
            payables=_f(r.get("balAccp")),
            goodwill_intangibles=_f(r.get("balGint")),
            retained_earnings=_f(r.get("balRtne")),
            long_term_debt=_f(r.get("balTltd")),
            total_liabilities=_f(r.get("balTotl")),
            minority_interest=_f(r.get("balMint")),
        )
        for r in _rows(payload)
    ][-YEARS_SHOWN:]


def _cashflow_periods(payload: Any) -> list[CashflowPeriod]:
    """Parses the cash flow endpoint into the periods shown."""
    return [
        CashflowPeriod(
            period=str(r.get("displayPeriod") or ""),
            cfo=_f(r.get("cafCfoa")),
            capex=_f(r.get("cafCexp")),
            fcf=_f(r.get("cafFcf")),
            dividends_paid=_f(r.get("cafTcdp")),
            change_in_working_capital=_f(r.get("cafCiwc")),
            cash_from_financing=_f(r.get("cafCffa")),
            cash_from_investing=_f(r.get("cafCfia")),
            net_change_in_cash=_f(r.get("cafNcic")),
        )
        for r in _rows(payload)
    ][-YEARS_SHOWN:]


def _apply_ratios(snap: CompanySnapshot, payloads: dict[str, Any]) -> None:
    """Fills the five ratio endpoints the provider computes itself."""
    snap.dupont = _ratio_block(payloads.get("dupont"), "dupont_5_factor")
    snap.solvency = _ratio_block(payloads.get("solvency"), "solvency_ratios")
    snap.liquidity = _ratio_block(payloads.get("liquidity"), "liquidity_ratios")
    snap.capital_efficiency = _ratio_block(
        payloads.get("capital_efficiency"), "capital_efficiency"
    )
    snap.cagr = _ratio_block(payloads.get("cagr"), "cagr_metrics")


def build_snapshot(ticker: str, payloads: dict[str, Any]) -> CompanySnapshot:
    """Assembles a CompanySnapshot from raw collector payloads.

    Args:
        ticker: Stock ticker symbol.
        payloads: Mapping from `CollectorClient.fetch_all`.

    Returns:
        A populated snapshot. Endpoints that failed leave their fields
        empty and append an entry to `warnings` rather than raising, so a
        partial report still renders with honest gaps.
    """
    snap = CompanySnapshot(ticker=ticker.upper())

    _apply_income(snap, payloads)

    snap.balance = _balance_periods(payloads.get("balance"))
    if not snap.balance:
        snap.warnings.append("balance sheet unavailable")

    snap.cashflow = _cashflow_periods(payloads.get("cashflow"))
    if not snap.cashflow:
        snap.warnings.append("cash flow statement unavailable")

    _apply_ratios(snap, payloads)

    summary = payloads.get("summary")
    if isinstance(summary, dict):
        _apply_summary(snap, summary)
    else:
        snap.warnings.append("summary profile unavailable")

    if not snap.peers:
        _apply_peer_fallback(snap, payloads.get("peers"))

    _validate(snap)
    return snap


def _apply_summary(snap: CompanySnapshot, summary: dict[str, Any]) -> None:
    """Populates profile, ratios, peers, holdings and dividends."""
    about = summary.get("aboutAndPeers")
    if isinstance(about, list) and about:
        subject = about[0] if isinstance(about[0], dict) else {}
        snap.name = str(subject.get("name") or snap.ticker)
        snap.sector = str(subject.get("sector") or "")
        snap.description = str(subject.get("description") or "")

        for item in about:
            if not isinstance(item, dict):
                continue
            ratios = item.get("ratios") if isinstance(item.get("ratios"), dict) else {}
            cap_mn = _f(ratios.get("marketCap"))
            snap.peers.append(
                Peer(
                    ticker=str(item.get("ticker") or ""),
                    name=str(item.get("name") or ""),
                    # Peer caps arrive in rupee million; everything else is crore.
                    market_cap_cr=(cap_mn / MN_PER_CR) if cap_mn is not None else None,
                    pe=_f(ratios.get("ttmPe")),
                    pb=_f(ratios.get("pbr")),
                    div_yield=_f(ratios.get("divDps")),
                    change_52w=_f(ratios.get("52wpct")),
                    buy_reco_pct=_f(ratios.get("breco")),
                    is_subject=str(item.get("ticker") or "").upper() == snap.ticker,
                )
            )

    for entry in summary.get("keyRatios") or []:
        if isinstance(entry, dict) and entry.get("backL") is not None:
            value = _f(entry.get("value"))
            if value is not None:
                snap.key_ratios[str(entry["backL"])] = value

    forecast = summary.get("forecast")
    if isinstance(forecast, dict):
        total = _f(forecast.get("totalReco"))
        snap.total_reco = int(total) if total is not None else None
        snap.buy_reco_pct = _f(forecast.get("percBuyReco"))

    holdings_root = summary.get("holdings")
    entries = holdings_root.get("holdings") if isinstance(holdings_root, dict) else None
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        data = entry.get("data") if isinstance(entry.get("data"), dict) else {}
        snap.holdings.append(
            Holding(
                date=str(entry.get("date") or "")[:10],
                promoter=_f(data.get("pmPctT")),
                fii=_f(data.get("fiPctT")),
                dii=_f(data.get("diPctT")),
                mutual_fund=_f(data.get("mfPctT")),
                insider=_f(data.get("isPctT")),
                retail_other=_f(data.get("rOthPctT")),
            )
        )
    # The provider returns holdings oldest-first, which is already the order
    # charts and tables need. Verified against the payload; do not reverse.

    dividends_root = summary.get("dividends")
    past = dividends_root.get("past") if isinstance(dividends_root, dict) else None
    for entry in past or []:
        if isinstance(entry, dict):
            snap.dividends.append(
                Dividend(
                    ex_date=str(entry.get("exDate") or "")[:10],
                    amount=_f(entry.get("dividend")),
                    kind=str(entry.get("subType") or ""),
                )
            )

    subject = next((p for p in snap.peers if p.is_subject), None)
    if subject is not None:
        snap.market_cap_cr = subject.market_cap_cr


def _apply_peer_fallback(snap: CompanySnapshot, payload: Any) -> None:
    """Populates peers from /peers when the summary carried none.

    The dedicated peers endpoint returns identity only — no ratios — so
    this yields names without a valuation grid.
    """
    root = payload.get("peers") if isinstance(payload, dict) else payload
    for item in root or []:
        if isinstance(item, dict) and item.get("ticker"):
            snap.peers.append(
                Peer(
                    ticker=str(item["ticker"]),
                    name=str(item.get("name") or ""),
                    is_subject=str(item["ticker"]).upper() == snap.ticker,
                )
            )
    if snap.peers:
        snap.warnings.append("peer valuation ratios unavailable; identities only")


def _validate(snap: CompanySnapshot) -> None:
    """Cross-checks derived figures and records any disagreement.

    The share count is the one figure available from two independent
    routes — the balance sheet's `balTcso`, and PAT divided by EPS. When
    they disagree materially, something upstream has changed and the
    per-share figures in the report cannot be trusted.
    """
    latest = snap.latest_balance
    reference = snap.ttm or (snap.years[-1] if snap.years else None)
    if latest is None or reference is None:
        return
    if latest.shares_cr is None or not reference.pat or not reference.eps:
        return

    implied = reference.pat / reference.eps
    reported = latest.shares_cr
    if reported <= 0:
        return
    drift = abs(implied - reported) / reported * 100.0
    if drift > 10.0:
        snap.warnings.append(
            f"share count disagrees by {drift:.1f}% (balance sheet {reported:.1f} cr vs "
            f"PAT/EPS {implied:.1f} cr); per-share figures unreliable"
        )
        logger.warning("[%s] share count drift %.1f%%", snap.ticker, drift)
