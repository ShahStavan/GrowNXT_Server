"""Durable graph orchestration for the Nifty 50 batch.

`ingestion.stages` holds the work and `ingestion.batch.process_ticker`
composes it for one ticker. This package adds the layer above: the run's own
loop, wired as a LangGraph `StateGraph` whose every superstep is written to
disk, so a run killed at ticker forty resumes there instead of re-visiting the
thirty-nine before it.

Four modules, deliberately separable:

* `state` -- the channel schema and its reducer. No IO, no langgraph import.
* `checkpoint` -- `JsonFileSaver`, a file-backed `BaseCheckpointSaver` that
  knows nothing about tickers and could be lifted into another project.
* `nodes` -- one adapter per node, plus the resources they are bound to.
* `build` -- the topology and its policies.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

from ingestion.graph.build import LANGGRAPH_AVAILABLE, build_batch_graph
from ingestion.graph.checkpoint import (
    CHECKPOINT_DIR_NAME,
    JsonFileSaver,
    checkpoint_root,
)
from ingestion.graph.nodes import GraphResources
from ingestion.graph.state import BatchGraphState, merge_results

__all__ = [
    "CHECKPOINT_DIR_NAME",
    "LANGGRAPH_AVAILABLE",
    "BatchGraphState",
    "GraphResources",
    "JsonFileSaver",
    "build_batch_graph",
    "checkpoint_root",
    "merge_results",
]
