"""Completion notification for batch jobs.

A batch that runs for hours, detached from any terminal, has to say once when
it is done -- and say so whether it finished cleanly or not; silence is only
acceptable for a run still in progress. The channels are deliberately few:

* the structured log line, always;
* one HTTP ``POST`` to a webhook, when ``NIFTY50_NOTIFY_WEBHOOK_URL`` (or an
  explicit URL) is set. Two wire formats, chosen from the URL:

  - **ntfy** (``https://ntfy.sh/<topic>`` or any host containing ``ntfy``):
    the summary as a plain-text body with ``Title``/``Tags``/``Priority``
    headers -- ntfy's own protocol, which needs no account. When
    ``NIFTY50_NOTIFY_EMAIL_TO`` is set (it defaults to the project author),
    an ``Email`` header makes ntfy forward the same message to that inbox, so
    one POST reaches both a phone and an email address.
  - **generic JSON** (everything else): a body carrying a Slack/Discord/Teams-
    compatible ``text`` field beside the machine-readable summary.

`notify_ticker` adds a second, quieter cadence on the same topic: one message
as each constituent finishes, so a detached fifty-ticker run is watchable from
a phone instead of only announcing itself at the end. It differs from the
summary by design -- low ntfy priority unless the ticker failed, and never an
``Email`` header, because the inbox wants the one summary and not the feed.

Delivery is best-effort with one retry. A webhook that is down is logged and
never raised: the run's recorded status is the source of truth, and a
notification failure must not be allowed to change it.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)

WEBHOOK_ENV: str = "NIFTY50_NOTIFY_WEBHOOK_URL"
EMAIL_TO_ENV: str = "NIFTY50_NOTIFY_EMAIL_TO"
# The project author (pyproject.toml); NIFTY50_NOTIFY_EMAIL_TO overrides, "" disables.
DEFAULT_EMAIL_TO: str = "shahstavan72@gmail.com"
WEBHOOK_TIMEOUT: float = 10.0
WEBHOOK_ATTEMPTS: int = 2

# ntfy tag shortcodes render as emoji in the app; priority 4 buzzes the phone.
NTFY_TAGS: dict[str, str] = {
    "COMPLETED": "white_check_mark,chart_with_upwards_trend",
    "COMPLETED_WITH_ERRORS": "warning,chart_with_upwards_trend",
    "FAILED": "x,rotating_light",
}
NTFY_PRIORITY: dict[str, str] = {
    "COMPLETED": "3",
    "COMPLETED_WITH_ERRORS": "4",
    "FAILED": "4",
}


@dataclass
class NotifyResult:
    """The outcome of one channel's delivery attempt."""

    channel: str
    ok: bool
    detail: str = ""


def webhook_url_from_env() -> str | None:
    """Returns the configured webhook URL, or None when notifications are off."""
    value = os.getenv(WEBHOOK_ENV, "").strip()
    return value or None


def email_to_from_env() -> str:
    """Returns the ntfy email-forwarding address; empty string disables it."""
    raw = os.getenv(EMAIL_TO_ENV)
    if raw is None:
        return DEFAULT_EMAIL_TO
    return raw.strip()


def is_ntfy_url(url: str) -> bool:
    """Whether a webhook URL points at an ntfy server (public or self-hosted)."""
    host = (urlsplit(url).hostname or "").lower()
    return host == "ntfy.sh" or host.endswith(".ntfy.sh") or "ntfy" in host


def _ascii(value: str, limit: int = 250) -> str:
    """HTTP headers are latin-1; keep ntfy headers plain ASCII and short."""
    return value.encode("ascii", "replace").decode("ascii")[:limit]


def format_ntfy_message(summary: dict[str, Any]) -> tuple[str, str]:
    """Returns ``(title, body)`` for an ntfy notification.

    The body is the summary line followed by the failed tickers, if any, and
    where the logs are -- what an operator needs to decide whether to look
    further, in a form a phone renders without a template.
    """
    status = str(summary.get("status", "UNKNOWN"))
    counters = summary.get("counters") or {}
    total = int(summary.get("total_tickers", 0))
    ok = int(counters.get("tickers_ok", 0))
    scope = "Nifty 50" if total >= 50 else f"{total} ticker(s)"
    title = f"GrowNXT {scope} embeddings {status}: {ok}/{total} OK"

    lines = [format_summary_line(summary)]
    failed = [
        f"{sym} ({run.get('status')})"
        for sym, run in sorted((summary.get("tickers") or {}).items())
        if str(run.get("status")) in ("FAILED", "PARTIAL")
    ]
    if failed:
        lines.append(
            "Needs attention: "
            + ", ".join(failed[:15])
            + (" ..." if len(failed) > 15 else "")
        )
    if summary.get("error"):
        lines.append(f"Run error: {summary['error']}")
    if summary.get("log_dir"):
        lines.append(f"Logs: {summary['log_dir']}")
    return title, "\n".join(lines)


def _post_ntfy_message(
    url: str,
    title: str,
    body: str,
    tags: str,
    priority: str,
    email_to: str = "",
    channel: str = "ntfy",
) -> NotifyResult:
    """POSTs one ntfy message, retrying a transient failure once.

    The transport both the end-of-run summary and the per-ticker progress
    messages share, so the retry policy and the 4xx/429 distinction are
    written once rather than diverging between the two.

    Args:
        url: The ntfy topic URL.
        title: Notification title; forced to ASCII, headers being latin-1.
        body: Plain-text body.
        tags: ntfy tag shortcodes, comma-separated.
        priority: ntfy priority, "1" (min) to "5" (max), as a string.
        email_to: When non-empty, an address ntfy also forwards to.
        channel: Name recorded on the returned result, for the run log.

    Returns:
        The delivery outcome. Never raises for an HTTP or network failure.
    """
    headers = {
        "Title": _ascii(title),
        "Tags": tags,
        "Priority": priority,
        "Content-Type": "text/plain; charset=utf-8",
    }
    if email_to:
        headers["Email"] = _ascii(email_to)

    last = ""
    for attempt in range(1, WEBHOOK_ATTEMPTS + 1):
        try:
            response = requests.post(
                url, data=body.encode("utf-8"), headers=headers, timeout=WEBHOOK_TIMEOUT
            )
            if response.ok:
                detail = f"HTTP {response.status_code}"
                if email_to:
                    detail += f", email -> {email_to}"
                return NotifyResult(channel, True, detail)
            last = f"HTTP {response.status_code}: {response.text[:120]}"
            if 400 <= response.status_code < 500 and response.status_code != 429:
                break
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "ntfy delivery attempt %d/%d failed: %s", attempt, WEBHOOK_ATTEMPTS, last
        )
    return NotifyResult(channel, False, last)


def _post_ntfy(url: str, summary: dict[str, Any], email_to: str) -> NotifyResult:
    """POSTs the end-of-run ntfy notification, forwarding to email when set."""
    status = str(summary.get("status", "UNKNOWN"))
    title, body = format_ntfy_message(summary)
    return _post_ntfy_message(
        url,
        title,
        body,
        NTFY_TAGS.get(status, "information_source"),
        NTFY_PRIORITY.get(status, "3"),
        email_to,
    )


def format_summary_line(summary: dict[str, Any]) -> str:
    """Renders the one-line human summary used for logs and chat webhooks."""
    status = str(summary.get("status", "UNKNOWN"))
    run_id = str(summary.get("run_id", ""))
    counters = summary.get("counters") or {}
    total = int(summary.get("total_tickers", 0))
    ok = int(counters.get("tickers_ok", 0))
    partial = int(counters.get("tickers_partial", 0))
    failed = int(counters.get("tickers_failed", 0))
    none = int(counters.get("tickers_no_documents", 0))
    embedded = int(counters.get("chunks_embedded", 0))
    new_docs = int(counters.get("documents_embedded", 0))
    elapsed = float(summary.get("elapsed_seconds", 0.0))
    if (summary.get("options") or {}).get("dry_run"):
        pending = int(counters.get("documents_pending", 0))
        unchanged = int(counters.get("documents_unchanged", 0))
        return (
            f"Nifty 50 embedding dry run {run_id}: {status} - "
            f"{int(counters.get('tickers_dry_run', 0))}/{total} tickers checked, "
            f"{pending} documents pending, {unchanged} unchanged, {failed} tickers "
            f"failed, {none} without documents; {elapsed / 60.0:.1f} min."
        )
    return (
        f"Nifty 50 embedding run {run_id}: {status} - "
        f"{ok}/{total} tickers OK, {partial} partial, {failed} failed, "
        f"{none} without documents; {new_docs} documents / {embedded} chunks "
        f"embedded in {elapsed / 60.0:.1f} min."
    )


def _post_webhook(url: str, body: dict[str, Any]) -> NotifyResult:
    last = ""
    for attempt in range(1, WEBHOOK_ATTEMPTS + 1):
        try:
            response = requests.post(url, json=body, timeout=WEBHOOK_TIMEOUT)
            if response.ok:
                return NotifyResult("webhook", True, f"HTTP {response.status_code}")
            last = f"HTTP {response.status_code}"
            if 400 <= response.status_code < 500 and response.status_code != 429:
                break
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Webhook delivery attempt %d/%d failed: %s", attempt, WEBHOOK_ATTEMPTS, last
        )
    return NotifyResult("webhook", False, last)


def notify_completion(
    summary: dict[str, Any],
    webhook_url: str | None = None,
    enabled: bool = True,
    email_to: str | None = None,
) -> list[NotifyResult]:
    """Announces a finished run on every configured channel.

    Args:
        summary: The run summary (status, counters, failed tickers, ...).
        webhook_url: Explicit webhook; defaults to `WEBHOOK_ENV`.
        enabled: False suppresses the webhook (smoke tests); the log line is
            still written.
        email_to: ntfy email-forwarding address; defaults to `EMAIL_TO_ENV`.
            Only used for ntfy URLs.

    Returns:
        One result per channel attempted. Never raises.
    """
    line = format_summary_line(summary)
    results = [NotifyResult("log", True, line)]
    level = logging.INFO if summary.get("status") == "COMPLETED" else logging.WARNING
    logger.log(level, "%s", line)

    url = webhook_url or webhook_url_from_env()
    if not enabled or not url:
        return results

    if is_ntfy_url(url):
        recipient = email_to if email_to is not None else email_to_from_env()
        try:
            results.append(_post_ntfy(url, summary, recipient))
        except Exception as exc:  # noqa: BLE001 - notification must never take the run down
            logger.error("ntfy notification raised: %s", exc)
            results.append(NotifyResult("ntfy", False, str(exc)))
        return results

    failed_tickers = [
        sym
        for sym, run in (summary.get("tickers") or {}).items()
        if str(run.get("status")) in ("FAILED", "PARTIAL")
    ]
    body = {
        "text": line,
        "run_id": summary.get("run_id"),
        "status": summary.get("status"),
        "total_tickers": summary.get("total_tickers"),
        "completed_tickers": summary.get("completed_tickers"),
        "counters": summary.get("counters") or {},
        "failed_tickers": failed_tickers,
        "started_at": summary.get("started_at"),
        "finished_at": summary.get("finished_at"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
    }
    try:
        results.append(_post_webhook(url, body))
    except Exception as exc:  # noqa: BLE001 - notification must never take the run down
        logger.error("Webhook notification raised: %s", exc)
        results.append(NotifyResult("webhook", False, str(exc)))
    return results


# --- Per-ticker progress -----------------------------------------------------

# A run posts one of these per constituent, so a healthy Nifty 50 run is fifty
# notifications on the same topic as the summary. Priority 2 is ntfy's "low":
# the message arrives and is listed in the app without waking the phone, which
# is what makes a fifty-message progress feed tolerable. A ticker that ends
# FAILED or PARTIAL is the reason to be watching at all, so it goes out at 4
# and buzzes exactly as the run summary does.
TICKER_PRIORITY: dict[str, str] = {
    "OK": "2",
    "NO_DOCUMENTS": "3",
    "DRY_RUN": "1",
    "PARTIAL": "4",
    "FAILED": "4",
}
TICKER_TAGS: dict[str, str] = {
    "OK": "white_check_mark",
    "NO_DOCUMENTS": "grey_question",
    "DRY_RUN": "mag",
    "PARTIAL": "warning",
    "FAILED": "x",
}
# Errors listed in a per-ticker body before it is truncated. A ticker whose
# every document failed would otherwise push a wall of tracebacks to a phone.
TICKER_ERROR_LIMIT: int = 3


def format_ticker_line(run: dict[str, Any], position: int = 0, total: int = 0) -> str:
    """Renders one finished ticker as a single human-readable line.

    Args:
        run: A `ingestion.batch.TickerRun.to_dict()` record.
        position: The ticker's 1-based position in the run, or 0 to omit it.
        total: Tickers in the run, or 0 to omit the position entirely.

    Returns:
        One line naming the symbol, its status and what it embedded.
    """
    symbol = str(run.get("symbol", "?"))
    status = str(run.get("status", "UNKNOWN"))
    where = f"[{position}/{total}] " if position and total else ""
    return (
        f"{where}{symbol} {status}: "
        f"{int(run.get('new_documents', 0))} new, "
        f"{int(run.get('unchanged_documents', 0))} unchanged, "
        f"{int(run.get('failed_documents', 0))} failed documents; "
        f"{int(run.get('chunks_indexed', 0))} chunks -> "
        f"{int(run.get('points_total', 0))} points in "
        f"{run.get('collection') or '-'}; "
        f"{float(run.get('elapsed_seconds', 0.0)) / 60.0:.1f} min."
    )


def format_ticker_message(
    run: dict[str, Any], position: int = 0, total: int = 0
) -> tuple[str, str]:
    """Returns ``(title, body)`` for one ticker's progress notification.

    The title carries the symbol, its verdict and the run's progress, because
    on a phone the title is the whole of what a glance reads. The body adds
    the document counts and, when something went wrong, the first few errors.

    Args:
        run: A `ingestion.batch.TickerRun.to_dict()` record.
        position: The ticker's 1-based position in the run.
        total: Tickers in the run.

    Returns:
        The notification title and its plain-text body.
    """
    symbol = str(run.get("symbol", "?"))
    status = str(run.get("status", "UNKNOWN"))
    progress = f" ({position}/{total})" if position and total else ""
    title = f"GrowNXT {symbol} {status}{progress}"

    lines = [format_ticker_line(run, position, total)]
    if run.get("name"):
        lines.insert(0, str(run["name"]))
    errors = [e for e in (run.get("errors") or []) if isinstance(e, dict)]
    for err in errors[:TICKER_ERROR_LIMIT]:
        stage = err.get("stage") or err.get("document") or "error"
        lines.append(f"{stage}: {str(err.get('error', ''))[:160]}")
    if len(errors) > TICKER_ERROR_LIMIT:
        lines.append(f"... and {len(errors) - TICKER_ERROR_LIMIT} more error(s)")
    return title, "\n".join(lines)


def notify_ticker(
    run: dict[str, Any],
    position: int = 0,
    total: int = 0,
    webhook_url: str | None = None,
    enabled: bool = True,
) -> NotifyResult | None:
    """Announces one finished ticker, mid-run.

    Deliberately unlike `notify_completion` in two ways. It never sets the
    ``Email`` header: fifty filings finishing would be fifty emails, and the
    inbox wants the one summary at the end, not the feed. And it returns a
    single result rather than a list, because there is no log channel here --
    `ingestion.runlog` has already recorded the ticker in `events.jsonl` by
    the time this is called, and duplicating it would only double the lines.

    Args:
        run: A `ingestion.batch.TickerRun.to_dict()` record.
        position: The ticker's 1-based position in the run.
        total: Tickers in the run.
        webhook_url: Explicit webhook; defaults to `WEBHOOK_ENV`.
        enabled: False suppresses the POST entirely.

    Returns:
        The delivery outcome, or None when no notification was attempted.
        Never raises: a ticker's result must not depend on a webhook.
    """
    url = webhook_url or webhook_url_from_env()
    if not enabled or not url:
        return None

    status = str(run.get("status", "UNKNOWN"))
    try:
        if is_ntfy_url(url):
            title, body = format_ticker_message(run, position, total)
            return _post_ntfy_message(
                url,
                title,
                body,
                TICKER_TAGS.get(status, "information_source"),
                TICKER_PRIORITY.get(status, "3"),
                email_to="",
                channel="ntfy_ticker",
            )
        body_json = {
            "text": format_ticker_line(run, position, total),
            "symbol": run.get("symbol"),
            "status": status,
            "position": position,
            "total_tickers": total,
            "new_documents": run.get("new_documents"),
            "unchanged_documents": run.get("unchanged_documents"),
            "failed_documents": run.get("failed_documents"),
            "chunks_indexed": run.get("chunks_indexed"),
            "points_total": run.get("points_total"),
            "collection": run.get("collection"),
            "elapsed_seconds": run.get("elapsed_seconds"),
        }
        result = _post_webhook(url, body_json)
        return NotifyResult("webhook_ticker", result.ok, result.detail)
    except Exception as exc:  # noqa: BLE001 - progress must never take a ticker down
        logger.error(
            "Per-ticker notification raised for %s: %s", run.get("symbol"), exc
        )
        return NotifyResult("ntfy_ticker", False, str(exc))
