"""The parity gate for the vectorless-qualitative-rag replacement.

Phase 0 of `.claude/plans/vectorless-qualitative-rag.md`. Every phase of that
plan replaces or deletes part of the retrieval stack, and each one has to leave
the rest of the tree working. This script is what "leaves the rest working"
means, in one command.

Three kinds of gate, in ascending cost:

1. **Import sweeps.** `ingestion/__init__.py` re-exports the entire surface
   being deleted, so a partial removal breaks it first (plan F5). Scripts are
   swept separately because `import ingestion` does not reach them -- the gap
   that hid two already-broken scripts until a full-tree grep found them
   (plan F10).
2. **Lint and format**, because the repository is checked in formatted and a
   stop hook enforces it.
3. **Verification suites**, which are the real behavioural gate and also the
   slow part.

Known-broken modules are declared in `KNOWN_BROKEN` rather than omitted, so
the gate reports them as expected failures. A module that starts importing
again is reported too: the plan deletes those two files, and silently passing
on them would hide the deletion never happening.

Google Python Style Guide Compliant.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import cli  # noqa: E402
from scripts.checks import Report, banner  # noqa: E402

# Packages whose import must keep working through every phase. The first is
# the one that actually catches an incomplete deletion.
PACKAGES: tuple[str, ...] = (
    "ingestion",
    "ingestion.rag",
    "ingestion.graph",
    "reporting",
    "core.config",
    "api.app",
    "app",
)

# Entry points in `scripts/`. Not reachable from `import ingestion`, which is
# exactly why they get their own sweep.
SCRIPTS: tuple[str, ...] = (
    "scripts.embed_nifty50",
    "scripts.generate_report",
    "scripts.evaluate_full_pipeline",
    "scripts.verify_nifty50_embeddings",
    "scripts.verify_documents",
    "scripts.verify_chunker",
    "scripts.verify_reporting",
    "scripts.verify_gdrive",
)

# Modules already broken before this work started, with the error each raises.
# CLAUDE.md documents both; the plan deletes them in Phase 5 (F10).
KNOWN_BROKEN: dict[str, str] = {
    "scripts.ingest_documents": "ImportError",
    "scripts.verify_ingestion": "ModuleNotFoundError",
}

# Verification suites, slowest last. Skipped by `--quick`.
SUITES: tuple[str, ...] = (
    "scripts/verify_reporting.py",
    "scripts/verify_documents.py",
    "scripts/verify_chunker.py",
    "scripts/verify_nifty50_embeddings.py",
)

LINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ruff format", ("-m", "ruff", "format", "--check", ".")),
    ("ruff check", ("-m", "ruff", "check", ".")),
)


def _import_ok(module: str) -> tuple[bool, str]:
    """Imports a module, returning whether it succeeded and the error if not.

    Args:
        module: Dotted module path.

    Returns:
        ``(ok, detail)``; `detail` is the exception type and message when the
        import failed, and empty otherwise.
    """
    try:
        importlib.import_module(module)
    except BaseException as exc:  # noqa: BLE001 - a gate reports, never raises
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _run(args: tuple[str, ...], timeout: int = 900) -> tuple[bool, str]:
    """Runs a Python subprocess and reports its outcome.

    A subprocess rather than an in-process call so that one suite's
    `sys.exit`, logging configuration or module-level state cannot affect the
    next one.

    Args:
        args: Arguments after the interpreter.
        timeout: Seconds before the run is abandoned.

    Returns:
        ``(ok, detail)`` where `detail` is the last meaningful output line on
        failure.
    """
    try:
        proc = subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"

    if proc.returncode == 0:
        return True, ""
    return False, _failure_detail(proc.stdout + proc.stderr, proc.returncode)


def _failure_detail(output: str, returncode: int) -> str:
    """Extracts the meaningful failure line from a suite's output.

    The naive answer -- the last non-empty line -- reports a tqdm progress bar
    from a model load, because that is what the suites print last. So look for
    what `scripts.checks.Report.finish` actually writes, then for a traceback's
    final line, and only then fall back to the tail.

    Args:
        output: Combined stdout and stderr.
        returncode: The process exit code, for the last-resort message.

    Returns:
        A single line, truncated for the listing.
    """
    lines = [ln.rstrip() for ln in output.splitlines() if ln.strip()]
    if not lines:
        return f"exit {returncode}"

    summary = next((ln for ln in reversed(lines) if ln.startswith("FAILED:")), "")
    named = [ln.strip() for ln in lines if ln.lstrip().startswith("- ")]
    if summary:
        first = f" | first: {named[0][2:]}" if named else ""
        return f"{summary}{first}"[:200]

    error = next(
        (
            ln.strip()
            for ln in reversed(lines)
            if "Error" in ln or ln.startswith("Traceback")
        ),
        "",
    )
    return (error or lines[-1])[:200]


def check_imports(report: Report) -> None:
    """Sweeps the package and script imports."""
    report.section("Import sweep - packages (plan F5)")
    for module in PACKAGES:
        ok, detail = _import_ok(module)
        report.check(module, ok, detail)

    report.section("Import sweep - scripts (plan F10)")
    for module in SCRIPTS:
        ok, detail = _import_ok(module)
        report.check(module, ok, detail)

    report.section("Import sweep - known broken, pending deletion (plan F10)")
    for module, expected in KNOWN_BROKEN.items():
        ok, detail = _import_ok(module)
        if ok:
            report.check(
                f"{module} still broken as recorded",
                False,
                "now imports cleanly - update KNOWN_BROKEN or delete the file",
            )
        else:
            report.check(
                f"{module} broken with {expected}",
                detail.startswith(expected),
                detail,
            )


def check_lint(report: Report) -> None:
    """Runs the formatter and linter in check mode."""
    report.section("Lint and format")
    for name, args in LINTS:
        ok, detail = _run(args, timeout=300)
        report.check(name, ok, detail)


def check_suites(report: Report, suites: tuple[str, ...]) -> None:
    """Runs the verification suites, reporting each one's wall time."""
    report.section("Verification suites")
    for suite in suites:
        if not (ROOT / suite).exists():
            report.check(suite, False, "missing")
            continue
        started = time.perf_counter()
        ok, detail = _run((suite,))
        elapsed = time.perf_counter() - started
        report.check(suite, ok, detail or f"{elapsed:.1f}s")


def main(argv: list[str] | None = None) -> int:
    """Runs the gate and returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Parity gate for the vectorless-qualitative-rag branch."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Imports and lint only; skip the verification suites.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        metavar="PATH",
        help="Run only this suite. Repeatable.",
    )
    args = parser.parse_args(argv)

    cli.console_utf8()
    banner("Parity gate - vectorless-qualitative-rag")
    report = Report()

    check_imports(report)
    check_lint(report)
    if not args.quick:
        check_suites(report, tuple(args.suite) if args.suite else SUITES)

    print(report.render())
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
