"""A file-backed LangGraph checkpoint saver.

`InMemorySaver` dies with the process, which is precisely the failure this
pipeline is being hardened against, and the SQLite and Postgres savers live in
packages this project does not install -- deliberately, because every other
piece of durable state here (``state.json``, ``run_manifest.json``,
``vectors.npz``) is file-backed and `CLAUDE.md` rules out adding a database
engine. So: one directory per thread, one JSON file per checkpoint, written
with the same temp-file-then-rename dance as `save_ticker_state`.

`JsonFileSaver` knows nothing about tickers, filings or the batch. It
implements the four methods `BaseCheckpointSaver` leaves abstract -- `put`,
`get_tuple`, `put_writes`, `list` -- plus `delete_thread`, and inherits the
rest. `get_delta_channel_history`, `prune`, `delete_for_runs` and
`copy_thread` are deliberately **not** overridden: the base implementations
work against the public contract, and the first of those is flagged beta
upstream.

Two departures from the in-memory reference, both for simplicity at this
scale:

* **Channel values are stored inline** with their checkpoint rather than as
  version-keyed blobs. The blob split exists to share unchanged values between
  checkpoints; graph state here is identifiers and paths by design
  (`ingestion.graph.state`), so the dedup buys little and costs a second file
  namespace and a version-keyed lookup.
* **There is no ``latest`` pointer file.** Checkpoint ids are UUIDv6 and
  therefore sort lexicographically by creation time, which is the same
  assumption the reference saver makes when it takes ``max(keys)``, so the
  newest checkpoint is found by scanning the directory. One fewer file to keep
  consistent.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import re
import shutil
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from core.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

# Where checkpoints live, under the cross-ticker state directory rather than
# any stock's workspace. `index_all_tickers` already skips `_`-prefixed
# directories, so nothing mistakes this for a constituent.
MANIFEST_DIR_NAME: str = "_nifty50"
CHECKPOINT_DIR_NAME: str = "checkpoints"

CHECKPOINT_SUFFIX: str = ".json"
WRITES_SUFFIX: str = ".writes.jsonl"
ROOT_NS_DIR: str = "__root__"

# Path segments are derived from a thread id and a checkpoint namespace, and a
# namespace for a nested subgraph looks like ``ticker:TCS|doc:annual_report``.
# Both ``:`` and ``|`` are illegal in Windows filenames, so the segment is
# slugified for the filesystem and the true value is recorded inside the file.
# The digest suffix is what keeps two namespaces that slugify alike apart.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_DIGEST_CHARS: int = 8


def checkpoint_root(output_dir: Path | None = None) -> Path:
    """Returns the directory holding every run's checkpoints.

    Args:
        output_dir: Artefact root. Defaults to `OUTPUT_DIR`.

    Returns:
        ``<output_dir>/_nifty50/checkpoints``. Not created here.
    """
    base = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    return base / MANIFEST_DIR_NAME / CHECKPOINT_DIR_NAME


def _slug(value: str, fallback: str) -> str:
    """Returns a filesystem-safe directory name for an arbitrary identifier."""
    if not value:
        return fallback
    cleaned = _UNSAFE.sub("_", value).strip("._") or fallback
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]
    return f"{cleaned[:60]}-{digest}"


def _encode(blob: tuple[str, bytes]) -> dict[str, str]:
    """Renders a serializer's ``(type, bytes)`` pair as JSON-safe fields."""
    kind, data = blob
    return {"type": kind, "data": base64.b64encode(data).decode("ascii")}


def _decode(payload: dict[str, Any]) -> tuple[str, bytes]:
    """Rebuilds a ``(type, bytes)`` pair from `_encode`'s output."""
    return str(payload["type"]), base64.b64decode(payload["data"])


class JsonFileSaver(BaseCheckpointSaver[int]):
    """Persists LangGraph checkpoints as JSON files under one directory.

    Layout, rooted at `checkpoint_root`::

        <thread_dir>/<ns_dir>/<checkpoint_id>.json
        <thread_dir>/<ns_dir>/<checkpoint_id>.writes.jsonl

    Thread-safe. `Send` fan-out runs sync nodes on a thread pool, so `put` and
    `put_writes` are called concurrently; a single lock around the write path
    is ample next to the cost of a Docling pass.

    Attributes:
        root: Directory holding every thread's checkpoints.
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        output_dir: Path | None = None,
        serde: Any = None,
    ) -> None:
        """Builds a saver.

        Args:
            root: Explicit checkpoint directory. Overrides `output_dir`.
            output_dir: Artefact root, from which the checkpoint directory is
                derived. Ignored when `root` is given.
            serde: Serializer override. Defaults to the inherited
                `JsonPlusSerializer`.
        """
        super().__init__(serde=serde)
        self.root = Path(root) if root is not None else checkpoint_root(output_dir)
        self._lock = threading.Lock()

    # --- Paths ---------------------------------------------------------------

    def _ns_dir(self, thread_id: str, checkpoint_ns: str) -> Path:
        """Returns the directory holding one namespace's checkpoints."""
        return (
            self.root
            / _slug(str(thread_id), "__thread__")
            / _slug(checkpoint_ns, ROOT_NS_DIR)
        )

    def _checkpoint_path(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> Path:
        return self._ns_dir(thread_id, checkpoint_ns) / (
            checkpoint_id + CHECKPOINT_SUFFIX
        )

    def _writes_path(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> Path:
        return self._ns_dir(thread_id, checkpoint_ns) / (checkpoint_id + WRITES_SUFFIX)

    def _thread_dirs(self) -> Iterator[Path]:
        """Yields every thread directory currently on disk."""
        if not self.root.exists():
            return
        yield from (p for p in sorted(self.root.iterdir()) if p.is_dir())

    @staticmethod
    def _write_atomically(path: Path, text: str) -> None:
        """Writes a file so a crash mid-write cannot leave it half-written."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
        except OSError:
            with contextlib.suppress(OSError):
                temporary.unlink()
            raise

    # --- Reading -------------------------------------------------------------

    def _read_checkpoint(self, path: Path) -> dict[str, Any] | None:
        """Reads one checkpoint file, or None when it is unusable.

        Never raises. A checkpoint that cannot be parsed must read as absent so
        the graph starts fresh, rather than failing a run on a torn file that
        the atomic write should already have prevented.
        """
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Unusable checkpoint at %s: %s", path, exc)
            return None

    def _read_writes(self, path: Path) -> list[PendingWrite]:
        """Reads a checkpoint's pending writes in the order they were stored."""
        if not path.exists():
            return []
        writes: list[PendingWrite] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.warning("Unusable writes sidecar at %s: %s", path, exc)
            return []
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                writes.append(
                    (
                        str(record["task_id"]),
                        str(record["channel"]),
                        self.serde.loads_typed(_decode(record["value"])),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                # One bad line must not discard the rest of the superstep.
                logger.warning("Skipping unusable write in %s: %s", path, exc)
        return writes

    def _checkpoint_ids(self, folder: Path) -> list[str]:
        """Returns a namespace's checkpoint ids, newest first.

        UUIDv6 ids sort by creation time, which is the same property the
        in-memory saver relies on when it takes ``max(keys)``.
        """
        if not folder.exists():
            return []
        ids = [
            p.name[: -len(CHECKPOINT_SUFFIX)]
            for p in folder.iterdir()
            if p.is_file()
            and p.name.endswith(CHECKPOINT_SUFFIX)
            and not p.name.endswith(WRITES_SUFFIX)
        ]
        return sorted(ids, reverse=True)

    def _tuple_from(self, folder: Path, checkpoint_id: str) -> CheckpointTuple | None:
        """Assembles one `CheckpointTuple` from its files, or None if absent.

        The thread id and namespace come from the record rather than the path,
        because the path segments are slugged for the filesystem and are not
        reversible. That is also what lets `list` walk directories it was given
        no config for.
        """
        record = self._read_checkpoint(folder / (checkpoint_id + CHECKPOINT_SUFFIX))
        if record is None:
            return None
        thread_id = str(record.get("thread_id", ""))
        checkpoint_ns = str(record.get("checkpoint_ns", ""))
        try:
            checkpoint: Checkpoint = self.serde.loads_typed(
                _decode(record["checkpoint"])
            )
            metadata: CheckpointMetadata = self.serde.loads_typed(
                _decode(record["metadata"])
            )
            channel_values = {
                channel: self.serde.loads_typed(_decode(blob))
                for channel, blob in (record.get("channel_values") or {}).items()
            }
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Undecodable checkpoint %s: %s", checkpoint_id, exc)
            return None

        parent_id = record.get("parent_checkpoint_id")
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint={**checkpoint, "channel_values": channel_values},
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": str(parent_id),
                    }
                }
                if parent_id
                else None
            ),
            pending_writes=self._read_writes(folder / (checkpoint_id + WRITES_SUFFIX)),
        )

    # --- BaseCheckpointSaver -------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Returns the checkpoint named by `config`, or the thread's newest.

        Args:
            config: Must carry ``thread_id``; ``checkpoint_ns`` defaults to the
                root namespace and ``checkpoint_id`` to the newest available.

        Returns:
            The checkpoint tuple, or None when the thread has none.
        """
        configurable = config.get("configurable") or {}
        thread_id = str(configurable.get("thread_id", ""))
        if not thread_id:
            return None
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        folder = self._ns_dir(thread_id, checkpoint_ns)

        requested = get_checkpoint_id(config)
        if requested:
            found = self._tuple_from(folder, requested)
            # Honour the caller's own config when it named the checkpoint, so
            # any extra configurable keys it carries survive the round trip.
            return found._replace(config=config) if found is not None else None

        for checkpoint_id in self._checkpoint_ids(folder):
            found = self._tuple_from(folder, checkpoint_id)
            if found is not None:
                return found
        return None

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002 - the base class's name
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Yields checkpoints newest first, honouring every filter given.

        Args:
            config: Restricts to one thread, and to one namespace or
                checkpoint id when it names them. None walks every thread.
            filter: Metadata keys that must all match exactly.
            before: Yield only checkpoints older than this one's id.
            limit: Stop after this many.

        Yields:
            Matching checkpoint tuples.
        """
        configurable = (config or {}).get("configurable") or {}
        wanted_thread = str(configurable.get("thread_id", "")) or None
        wanted_ns = configurable.get("checkpoint_ns")
        wanted_id = get_checkpoint_id(config) if config else None
        before_id = get_checkpoint_id(before) if before else None
        remaining = limit

        for thread_dir in self._thread_dirs():
            for ns_dir in sorted(p for p in thread_dir.iterdir() if p.is_dir()):
                for checkpoint_id in self._checkpoint_ids(ns_dir):
                    if wanted_id and checkpoint_id != wanted_id:
                        continue
                    if before_id and checkpoint_id >= before_id:
                        continue
                    if remaining is not None and remaining <= 0:
                        return

                    found = self._tuple_from(ns_dir, checkpoint_id)
                    if found is None:
                        continue
                    found_conf = found.config["configurable"]
                    if wanted_thread and found_conf.get("thread_id") != wanted_thread:
                        continue
                    if (
                        wanted_ns is not None
                        and found_conf.get("checkpoint_ns") != wanted_ns
                    ):
                        continue
                    if filter and not all(
                        found.metadata.get(key) == value
                        for key, value in filter.items()
                    ):
                        continue

                    if remaining is not None:
                        remaining -= 1
                    yield found

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,  # noqa: ARG002 - values are stored inline
    ) -> RunnableConfig:
        """Writes a checkpoint and returns the config that addresses it.

        Args:
            config: Carries the thread, namespace, and the parent checkpoint id.
            checkpoint: The checkpoint to store.
            metadata: Metadata to store alongside it.
            new_versions: Channel versions new as of this write. Unused here:
                channel values are stored inline rather than as version-keyed
                blobs, so there is nothing to key by version.

        Returns:
            A config naming the checkpoint just written.
        """
        configurable = config.get("configurable") or {}
        thread_id = str(configurable.get("thread_id", ""))
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_id = str(checkpoint["id"])

        shallow = dict(checkpoint)
        values: dict[str, Any] = shallow.pop("channel_values", {}) or {}
        record = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": configurable.get("checkpoint_id"),
            "checkpoint": _encode(self.serde.dumps_typed(shallow)),
            "metadata": _encode(
                self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
            ),
            "channel_values": {
                channel: _encode(self.serde.dumps_typed(value))
                for channel, value in values.items()
            },
        }

        path = self._checkpoint_path(thread_id, checkpoint_ns, checkpoint_id)
        with self._lock:
            self._write_atomically(path, json.dumps(record, ensure_ascii=False))
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Appends one task's intermediate writes to its checkpoint's sidecar.

        Idempotent per ``(task_id, index)`` for ordinary writes, so a replayed
        superstep does not duplicate them. The special channels in
        `WRITES_IDX_MAP` map to negative indices and are allowed to overwrite,
        which is how an error or an interrupt replaces an earlier one.

        Args:
            config: Must name the thread, namespace and checkpoint.
            writes: ``(channel, value)`` pairs to store.
            task_id: The task producing them.
            task_path: The task's path within the graph.
        """
        configurable = config.get("configurable") or {}
        thread_id = str(configurable.get("thread_id", ""))
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_id = str(configurable.get("checkpoint_id", ""))
        if not checkpoint_id:
            return

        path = self._writes_path(thread_id, checkpoint_ns, checkpoint_id)
        with self._lock:
            seen = self._write_keys(path)
            pending: list[str] = []
            for index, (channel, value) in enumerate(writes):
                position = WRITES_IDX_MAP.get(channel, index)
                key = f"{task_id}:{position}"
                if position >= 0 and key in seen:
                    continue
                seen.add(key)
                pending.append(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "task_path": task_path,
                            "index": position,
                            "channel": channel,
                            "value": _encode(self.serde.dumps_typed(value)),
                        },
                        ensure_ascii=False,
                    )
                )
            if not pending:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(pending) + "\n")

    def _write_keys(self, path: Path) -> set[str]:
        """Returns the ``task_id:index`` keys already stored in a sidecar."""
        if not path.exists():
            return set()
        keys: set[str] = set()
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                with contextlib.suppress(ValueError, KeyError):
                    record = json.loads(line)
                    keys.add(f"{record['task_id']}:{record['index']}")
        except OSError as exc:
            logger.warning("Could not read write keys from %s: %s", path, exc)
        return keys

    def delete_thread(self, thread_id: str) -> None:
        """Removes every checkpoint and write belonging to one thread.

        Args:
            thread_id: The thread to forget. Absent threads are not an error.
        """
        folder = self.root / _slug(str(thread_id), "__thread__")
        with self._lock:
            shutil.rmtree(folder, ignore_errors=True)

    # --- Async delegates -----------------------------------------------------
    #
    # The batch is synchronous, but `app.py` mounts an ASGI server: a future
    # async caller must not block the event loop on these file reads.

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Async `get_tuple`."""
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002 - the base class's name
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async `list`."""
        found = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in found:
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async `put`."""
        return await asyncio.to_thread(
            self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async `put_writes`."""
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        """Async `delete_thread`."""
        await asyncio.to_thread(self.delete_thread, thread_id)
