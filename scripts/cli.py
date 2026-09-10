"""Command-line plumbing shared by this package's entry points.

One place for logging setup and the stdout re-encoding, and three named
formats to choose between. `console_utf8` is not optional: a Windows
console defaults to cp1252 and raises on the first rupee sign.
"""

import contextlib
import logging
import sys
from typing import TextIO

# Level with no name, for a terse one-shot command.
PLAIN: str = "%(levelname)s %(message)s"
# Level and logger name, for output where the source module matters.
NAMED: str = "%(levelname)-7s %(name)s: %(message)s"
# Stamped, for a long run where elapsed time is the useful signal.
TIMED: str = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

CLOCK: str = "%H:%M:%S"


def console_utf8() -> None:
    """Makes stdout tolerate the characters filings actually contain.

    Filings carry rupee signs and typographic dashes. A Windows console
    defaults to cp1252 and raises on the first one, which would end a run
    that had otherwise succeeded.
    """
    if not hasattr(sys.stdout, "reconfigure"):
        return
    with contextlib.suppress(
        ValueError, OSError
    ):  # pragma: no cover - redirected stdout
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def setup(
    level: int = logging.INFO,
    fmt: str = PLAIN,
    stream: TextIO | None = None,
    datefmt: str | None = None,
) -> None:
    """Prepares the console and root logging for a command-line run.

    Args:
        level: Root logging level.
        fmt: One of `PLAIN`, `NAMED` or `TIMED`.
        stream: Where records go; stderr keeps them clear of piped output.
        datefmt: Timestamp format, for `TIMED`.

    """
    console_utf8()
    logging.basicConfig(level=level, format=fmt, stream=stream, datefmt=datefmt)
