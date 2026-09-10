"""File-backed logging for long-running batch jobs.

The console loggers in ``scripts/cli.py`` are right for a one-shot command and
wrong for a job that runs for hours, detached from any terminal: its record has
to outlive the shell that started it, and "how many documents were fetched"
has to be answerable without reading prose. So a batch run writes three
things under ``logs/<job>/``:

* ``<job>.log`` -- one rolling, human-readable log across every run;
* ``runs/<run_id>/run.log`` -- the full DEBUG log of one run;
* ``runs/<run_id>/events.jsonl`` -- one JSON object per line, one line per
  stage boundary (a catalogue fetched, a document downloaded, a chunk set
  embedded), carrying the counts and timings as fields rather than in text.

``runs/<run_id>/summary.json`` holds the run's final counters and
``latest.json`` points at the most recent run, so a status command needs
neither a terminal nor a log parser.

Handlers attach to the root logger, so records from ``ingestion.catalog``,
``ingestion.documents.*``, ``ingestion.chunker`` and ``ingestion.indexer`` are
captured without those modules knowing a batch job exists. Every file opens as
UTF-8: filings carry rupee signs and typographic dashes, and a handler that
raised on the first one would end a run that had otherwise succeeded.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading
from collections import Counter
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Root of all job logs. Override for container volumes.
_env_log_dir: str = os.getenv("GROWNXT_LOG_DIR", "").strip()
LOG_ROOT: Path = Path(_env_log_dir).resolve() if _env_log_dir else PROJECT_ROOT / "logs"

ROLLING_MAX_BYTES: int = 10 * 1024 * 1024
ROLLING_BACKUPS: int = 5

RUN_LOG_NAME: str = "run.log"
EVENTS_NAME: str = "events.jsonl"
SUMMARY_NAME: str = "summary.json"
LATEST_NAME: str = "latest.json"
STDOUT_NAME: str = "stdout.log"

TIMED_FORMAT: str = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
CLOCK: str = "%Y-%m-%d %H:%M:%S"


def utc_now() -> str:
    """Returns the current UTC time as an ISO-8601 string, to the second."""
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


def new_run_id(now: _dt.datetime | None = None) -> str:
    """Returns a sortable run identifier such as ``20260903T041500Z``."""
    stamp = now or _dt.datetime.now(_dt.UTC)
    return stamp.strftime("%Y%m%dT%H%M%SZ")


def _atomic_write_json(path: Path, data: Any) -> None:
    """Writes JSON via a temp file and ``os.replace`` so readers never see a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    temp.replace(path)


def _jsonable(value: Any) -> Any:
    """Coerces sets, paths and other non-JSON values for the event stream."""
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class RunLogger:
    """Owns one run's log files, event stream and counters.

    Thread-safe: a batch that processes tickers on several workers writes
    events and counters from each of them.

    Attributes:
        job: Job name; the sub-directory under the log root.
        run_id: This run's identifier.
        job_dir: ``<root>/<job>``.
        run_dir: ``<root>/<job>/runs/<run_id>``.
        counters: Run-level integer counters, incremented via `add`.
        timings: Seconds spent per stage, accumulated via `time_stage`.
    """

    def __init__(
        self,
        job: str,
        run_id: str,
        root: Path | None = None,
        verbose: bool = False,
    ) -> None:
        self.job = job
        self.run_id = run_id
        self.job_dir = (Path(root) if root is not None else LOG_ROOT) / job
        self.run_dir = self.job_dir / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.counters: Counter[str] = Counter()
        self.timings: Counter[str] = Counter()
        self._lock = threading.Lock()
        self._handlers: list[logging.Handler] = []
        self._events = (self.run_dir / EVENTS_NAME).open("a", encoding="utf-8")
        self._attach(verbose)

    # --- Paths ---------------------------------------------------------------

    @property
    def rolling_log(self) -> Path:
        """The job-wide rolling log."""
        return self.job_dir / f"{self.job}.log"

    @property
    def run_log(self) -> Path:
        """This run's DEBUG log."""
        return self.run_dir / RUN_LOG_NAME

    @property
    def events_path(self) -> Path:
        """This run's JSONL event stream."""
        return self.run_dir / EVENTS_NAME

    @property
    def summary_path(self) -> Path:
        """This run's final summary."""
        return self.run_dir / SUMMARY_NAME

    @property
    def stdout_path(self) -> Path:
        """Where a detached run's raw stdout/stderr is redirected."""
        return self.run_dir / STDOUT_NAME

    # --- Handlers ------------------------------------------------------------

    def _attach(self, verbose: bool) -> None:
        formatter = logging.Formatter(TIMED_FORMAT, datefmt=CLOCK)
        root = logging.getLogger()
        if root.level == logging.NOTSET or root.level > logging.DEBUG:
            root.setLevel(logging.DEBUG if verbose else logging.INFO)

        rolling = RotatingFileHandler(
            self.rolling_log,
            maxBytes=ROLLING_MAX_BYTES,
            backupCount=ROLLING_BACKUPS,
            encoding="utf-8",
        )
        rolling.setLevel(logging.INFO)
        rolling.setFormatter(formatter)

        per_run = logging.FileHandler(self.run_log, encoding="utf-8")
        per_run.setLevel(logging.DEBUG if verbose else logging.INFO)
        per_run.setFormatter(formatter)

        for handler in (rolling, per_run):
            root.addHandler(handler)
            self._handlers.append(handler)

    def close(self) -> None:
        """Detaches the file handlers and closes the event stream."""
        root = logging.getLogger()
        for handler in self._handlers:
            root.removeHandler(handler)
            handler.close()
        self._handlers.clear()
        with self._lock:
            if not self._events.closed:
                self._events.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # --- Events & counters ---------------------------------------------------

    def event(self, name: str, ticker: str = "", **fields: Any) -> dict[str, Any]:
        """Appends one structured event and returns it.

        Flushed per line so ``tail -f events.jsonl`` follows a live run.
        """
        record: dict[str, Any] = {"ts": utc_now(), "run_id": self.run_id, "event": name}
        if ticker:
            record["ticker"] = ticker
        record.update(_jsonable(fields))
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            if not self._events.closed:
                self._events.write(line + "\n")
                self._events.flush()
        logger.debug("event %s %s", name, ticker or "")
        return record

    def add(self, counter: str, amount: int = 1) -> None:
        """Increments a run-level counter."""
        with self._lock:
            self.counters[counter] += int(amount)

    def time_stage(self, stage: str, seconds: float) -> None:
        """Accumulates wall time against a pipeline stage."""
        with self._lock:
            self.timings[stage] += int(round(max(0.0, seconds) * 1000))

    def snapshot(self) -> dict[str, Any]:
        """Returns the counters and per-stage seconds as plain dicts."""
        with self._lock:
            return {
                "counters": dict(sorted(self.counters.items())),
                "elapsed_by_stage_seconds": {
                    k: round(v / 1000.0, 3) for k, v in sorted(self.timings.items())
                },
            }

    # --- Summary files -------------------------------------------------------

    def write_summary(self, data: dict[str, Any]) -> Path:
        """Writes the run's final summary, merged with the live counters."""
        payload = {**data, **self.snapshot()}
        _atomic_write_json(self.summary_path, payload)
        return self.summary_path

    def write_latest(self, status: str, **extra: Any) -> Path:
        """Points ``latest.json`` at this run."""
        path = self.job_dir / LATEST_NAME
        _atomic_write_json(
            path,
            {
                "run_id": self.run_id,
                "status": status,
                "updated_at": utc_now(),
                "run_dir": str(self.run_dir),
                "run_log": str(self.run_log),
                "events": str(self.events_path),
                "summary": str(self.summary_path),
                **_jsonable(extra),
            },
        )
        return path


def setup_run_logging(
    job: str,
    run_id: str | None = None,
    root: Path | None = None,
    verbose: bool = False,
) -> RunLogger:
    """Creates the run directory and attaches file logging for one run.

    Args:
        job: Job name, e.g. ``nifty50``.
        run_id: Explicit identifier; generated from the clock when absent.
        root: Log root; defaults to `LOG_ROOT`.
        verbose: Write DEBUG records to the per-run log.

    Returns:
        The run's `RunLogger`; call `close()` (or use as a context manager)
        when the run ends.
    """
    return RunLogger(job=job, run_id=run_id or new_run_id(), root=root, verbose=verbose)


def read_latest(job: str, root: Path | None = None) -> dict[str, Any] | None:
    """Returns the ``latest.json`` pointer for a job, or None when absent/corrupt."""
    path = (Path(root) if root is not None else LOG_ROOT) / job / LATEST_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None
