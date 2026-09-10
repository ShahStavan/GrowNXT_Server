"""Assembles the batch graph: topology and policies, no business logic.

The shape is a bounded loop rather than a single fan-out::

    START -> dispatch -> [ticker] x N -> collect -> dispatch -> ... -> END

`dispatch` takes `workers` symbols off the queue, a conditional edge sends one
task per symbol, `collect` records them, and the edge out of `collect` goes
back to `dispatch` until the queue is empty. Every arrow is a superstep and
every superstep is a checkpoint, so the queue on disk is what a resumed run
reads to know which tickers are left.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import logging
from typing import Any

from ingestion.graph.nodes import (
    COLLECT,
    DISPATCH,
    TICKER,
    GraphResources,
    collect,
    dispatch,
    route_collect,
    route_dispatch,
    run_ticker,
)
from ingestion.graph.state import BatchGraphState

logger = logging.getLogger(__name__)

try:  # pragma: no cover - langgraph is pinned in requirements
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - mirrors ingestion.rag.pipeline
    LANGGRAPH_AVAILABLE = False
    StateGraph = Any  # type: ignore[assignment,misc]
    START = "START"
    END = "END"

# How many times a ticker's whole pipeline is retried. One: `process_ticker`
# never raises -- it records failures against the document or ticker they
# belong to -- so a raised exception here is a bug in the orchestration, and
# retrying it would just repeat it.
#
# The narrower retries the spec calls for live where the transient failures
# actually are, and two of them are deliberately absent:
#
# * The downloader already retries with backoff and honours `Retry-After`
#   (`ingestion.documents.download.Downloader.fetch`), and tells a transient
#   failure from a permanent one. A retry wrapped around that would multiply
#   attempts against an exchange server -- the opposite of the politeness it
#   was written for. Do not add one.
# * A Docling failure on a malformed PDF is deterministic. Retrying costs
#   three full passes to reach the same error.
#
# Qdrant's transient 5xx is the one worth retrying, and that belongs inside
# `index_chunk_set`, per document, rather than around a whole ticker.


def build_batch_graph(res: GraphResources, checkpointer: Any = None) -> Any:
    """Compiles the batch graph.

    Args:
        res: The run's bound resources. Node functions close over this, so it
            carries everything that must not enter graph state.
        checkpointer: A `BaseCheckpointSaver`, or `False` to compile without
            one. `None` lets a parent graph's checkpointer be inherited, which
            for this top-level graph means no durability -- so callers that
            want none should pass `False` explicitly.

    Returns:
        The compiled graph.

    Raises:
        RuntimeError: If langgraph is not installed.
    """
    if not LANGGRAPH_AVAILABLE:  # pragma: no cover - dependency is pinned
        raise RuntimeError(
            "langgraph is required to build the batch graph; "
            "install it or run with --no-checkpoint"
        )

    builder = StateGraph(BatchGraphState)
    builder.add_node(DISPATCH, lambda state: dispatch(res, state))
    builder.add_node(TICKER, lambda state: run_ticker(res, state))
    builder.add_node(COLLECT, lambda state: collect(res, state))

    builder.add_edge(START, DISPATCH)
    # Node-level caching is deliberately not configured. `CachePolicy` keys on
    # a hash of the node's *input*, and only in-memory and Redis caches are
    # available -- neither survives the crash this graph exists to recover
    # from. The real cache is content-addressed and on disk: the PDF, the
    # extraction and the chunk file under `output/<TICKER>/`, keyed by the
    # filing's digest and the pipeline versions. See `Extractor._cached`.
    builder.add_conditional_edges(DISPATCH, route_dispatch, [TICKER, END])
    builder.add_edge(TICKER, COLLECT)
    builder.add_conditional_edges(COLLECT, route_collect, [DISPATCH, END])

    return builder.compile(checkpointer=checkpointer)
