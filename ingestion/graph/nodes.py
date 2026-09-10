"""The batch graph's nodes: adapters between graph state and the stages.

Each node reads the state, calls one thing, and returns a state delta. None of
them contains pipeline logic -- that lives in `ingestion.stages`, and the
per-ticker composition of it lives in `ingestion.batch.process_ticker`, which
is what `run_ticker` calls. A node that grew a loop or a second stage call
would be doing work the layer below it already does.

What the graph adds over calling `process_ticker` in a `for` loop is the
checkpoint between supersteps: the dispatch queue, the finished tickers and
the counters are on disk after every batch, so a run that dies at ticker forty
resumes at forty rather than re-visiting the thirty-nine before it.

**Why the per-document level is not a subgraph.** The spec sketched a nested
document subgraph so a checkpoint could land between one filing's extraction
and its chunking. That was written before the extraction and chunk caches
existed. With them, re-entering a partly-done ticker costs a few
cache-validating file reads rather than a Docling pass, so the checkpoint
would save seconds while adding a second state schema, an output-schema
mapping onto the parent, and a namespace per filing. The cost was moved, not
worked around; see `ingestion.documents.extract.Extractor._cached`.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ingestion.graph.state import BatchGraphState

logger = logging.getLogger(__name__)

# Node names, so the topology and the nodes cannot disagree about a string.
DISPATCH: str = "dispatch"
TICKER: str = "ticker"
COLLECT: str = "collect"


@dataclass(frozen=True)
class GraphResources:
    """The non-serialisable things every node needs, bound at build time.

    An indexer, a logger and a callback cannot live in graph state -- the
    checkpointer would have to serialise them, and `ingestion.graph.state`
    exists to keep exactly this kind of object out of it. So they are captured
    when the graph is built and passed to each node explicitly, which also
    means a node can be called directly in a test.

    Attributes:
        options: The run's `BatchOptions`.
        indexer: The vector store, shared across tickers.
        runlog: The run's logger, which owns the run-level counters.
        workers: Tickers to dispatch per superstep, already capped by
            `resolve_workers`.
        members: Constituents in scope, keyed by directory-safe symbol.
        order: Symbols in dispatch order, which is constituent order.
        record: Called with each finished `TickerRun`; this is what writes the
            run manifest and prints the operator's per-ticker row.
    """

    options: Any
    indexer: Any
    runlog: Any
    workers: int
    members: dict[str, Any]
    order: Sequence[str]
    record: Callable[[Any], None]

    @property
    def total(self) -> int:
        """Tickers in this run, for the per-ticker log line."""
        return len(self.order)


def dispatch(res: GraphResources, state: BatchGraphState) -> dict[str, Any]:
    """Takes the next batch of tickers off the queue.

    Concurrency is bounded here rather than by fanning out every ticker at
    once, because `Send` has no worker limit of its own: on a small GPU each
    concurrent ticker loads its own layout and table models onto the same
    card, which is the out-of-memory `resolve_workers` exists to prevent. A
    batch boundary is also a checkpoint, so resume granularity is at worst
    `workers` tickers.

    Args:
        res: The run's bound resources.
        state: The batch state.

    Returns:
        The shortened queue and the symbols now in flight.
    """
    queue = list(state.get("queue") or [])
    if not queue:
        return {"in_flight": []}
    size = max(1, int(state.get("workers") or res.workers))
    return {"queue": queue[size:], "in_flight": queue[:size]}


def route_dispatch(state: BatchGraphState) -> list[Any] | str:
    """Fans the in-flight batch out to `ticker`, or ends the run.

    Returns:
        One `Send` per ticker in the batch, or the END sentinel when the queue
        is exhausted.
    """
    from langgraph.graph import END
    from langgraph.types import Send

    in_flight = list(state.get("in_flight") or [])
    if not in_flight:
        return END
    done = len(state.get("results") or {})
    return [
        Send(TICKER, {"symbol": symbol, "position": done + offset})
        for offset, symbol in enumerate(in_flight, start=1)
    ]


def run_ticker(res: GraphResources, state: BatchGraphState) -> dict[str, Any]:
    """Runs one ticker's whole pipeline and returns its record.

    Never raises: `process_ticker` records every failure against the document
    or ticker it belongs to, and one bad constituent must not end the batch.

    Args:
        res: The run's bound resources.
        state: The `Send` payload -- a symbol and its position in the run.

    Returns:
        The ticker's `TickerRun`, keyed by symbol, for the `results` channel.
    """
    from ingestion.batch import process_ticker

    symbol = str(state.get("symbol") or "")
    member = res.members.get(symbol)
    if member is None:
        logger.error("dispatched an unknown symbol %r; skipping.", symbol)
        return {}
    if res.options.delay_seconds > 0 and int(state.get("position") or 0) > 1:
        import time

        time.sleep(res.options.delay_seconds)

    run = process_ticker(
        member,
        res.options,
        res.indexer,
        res.runlog,
        int(state.get("position") or 0),
        res.total,
    )
    # `to_dict` rather than the dataclass: what the checkpointer serialises
    # must be data, and `record` rebuilds the record it needs.
    return {"results": {run.symbol: run.to_dict()}}


def collect(res: GraphResources, state: BatchGraphState) -> dict[str, Any]:
    """Records the tickers this superstep finished.

    The manifest is written from here rather than at the end of the run, so a
    poller sees progress and a crash leaves behind what was already done.

    Args:
        res: The run's bound resources.
        state: The batch state, whose `results` now include this batch.

    Returns:
        The counters snapshot, and an empty in-flight list.
    """
    from ingestion.batch import TickerRun

    for symbol in state.get("in_flight") or []:
        recorded = (state.get("results") or {}).get(symbol)
        if recorded is None:
            continue
        with contextlib.suppress(Exception):
            # A printing or manifest-writing callback must never fail a
            # superstep: the ticker's work is already done and persisted.
            res.record(TickerRun.from_dict(recorded))
    return {
        "in_flight": [],
        "counters": dict(res.runlog.snapshot().get("counters") or {}),
    }


def route_collect(state: BatchGraphState) -> str:
    """Loops back for the next batch, or ends the run."""
    from langgraph.graph import END

    return DISPATCH if state.get("queue") else END
