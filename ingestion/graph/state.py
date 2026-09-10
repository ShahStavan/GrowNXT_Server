"""The batch graph's channel schema and its one reducer.

**The rule this module exists to enforce:** state carries identifiers, counts
and plain records -- never payloads. No `ExtractedDocument`, no `ChunkSet`, no
chunk text, no vectors, no PDF bytes, and no live object like an indexer or a
logger. A checkpoint is written on every superstep, so a state holding one
annual report's chunk set writes megabytes per node per ticker and the
checkpointer stops being an optimisation and becomes the bottleneck. Nodes
re-open artefacts from the `DocumentStore`, and `ingestion.graph.nodes`
receives its live objects through `GraphResources` instead.

`merge_results` is a module-level named function rather than a lambda because
`Send` fan-out means several ticker tasks fold into that channel in one
superstep, and the reducer has to be importable to be tested directly.

Deliberately free of any `langgraph` import: the schema is a plain `TypedDict`
and the reducer a plain function, so both can be exercised without a graph
runtime.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

__all__ = ["BatchGraphState", "merge_results"]


def merge_results(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> dict[str, Any]:
    """Folds finished per-ticker records together, keyed by symbol.

    The one channel several tasks write in the same superstep, so it needs a
    reducer rather than the default last-write-wins over the whole dict --
    without one, two tickers finishing together would leave only the second.
    Per *symbol*, last write does win, which is correct because each ticker is
    written by exactly one task: a collision can only be a replay of that
    ticker, whose later answer is the one to keep.

    Args:
        left: Records accumulated so far.
        right: Records from this superstep.

    Returns:
        A new dict; neither argument is mutated.
    """
    merged: dict[str, Any] = dict(left or {})
    merged.update(right or {})
    return merged


class BatchGraphState(TypedDict, total=False):
    """The run's state, checkpointed under ``thread_id = run_id``.

    Attributes:
        queue: Symbols not yet dispatched, in constituent order. Shrinks by a
            batch per superstep, and is what makes a resumed run pick up where
            it stopped instead of re-visiting finished tickers.
        in_flight: Symbols dispatched in the current superstep.
        results: Finished `TickerRun.to_dict()` per symbol. Plain dicts rather
            than the dataclass, so what the checkpointer serialises is data.
        counters: The run's counters as of the last superstep. A snapshot
            written by `collect`, not an accumulator: `ingestion.runlog` owns
            the counters and is already thread-safe, so the graph records them
            for the manifest rather than adding them up a second time.
        symbol: The ticker a `Send` addressed. Present only in the payload one
            `ticker` task receives, never in the run's own state.
        position: That ticker's 1-based position in the run, for the log.
    """

    queue: list[str]
    in_flight: list[str]
    results: Annotated[dict[str, Any], merge_results]
    counters: dict[str, int]
    symbol: str
    position: int
