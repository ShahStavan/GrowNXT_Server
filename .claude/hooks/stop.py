from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ruff_command() -> list[str]:
    """Locates a runnable `ruff`, preferring the project's own venv.

    A bare `"ruff"` only resolves when the venv's Scripts/bin directory is on
    PATH, which is not guaranteed for the process running this hook -- that
    raised FileNotFoundError instead of a lint result. The venv copy is
    known-installed (see requirements-dev.txt) and is what "ruff format ."
    should mean for this repo regardless of the invoking shell's PATH.
    """
    venv_ruff = PROJECT_ROOT / "venv" / "Scripts" / "ruff.exe"
    if venv_ruff.exists():
        return [str(venv_ruff)]

    venv_ruff_posix = PROJECT_ROOT / "venv" / "bin" / "ruff"
    if venv_ruff_posix.exists():
        return [str(venv_ruff_posix)]

    on_path = shutil.which("ruff")
    if on_path:
        return [on_path]

    return [sys.executable, "-m", "ruff"]


def run(args: list[str]) -> int:
    command = _ruff_command() + args
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        print(
            f"ERROR: could not run ruff ({' '.join(command)}): {exc}",
            file=sys.stderr,
        )
        return 127

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode


def main() -> int:
    print("Running final Ruff validation...")

    format_result = run(
        [
            "format",
            "--check",
            ".",
        ]
    )

    if format_result != 0:
        print(
            "ERROR: Ruff formatting check failed.",
            file=sys.stderr,
        )
        return 2

    lint_result = run(
        [
            "check",
            ".",
        ]
    )

    if lint_result != 0:
        print(
            "ERROR: Ruff linting check failed.",
            file=sys.stderr,
        )
        return 2

    print("Final Ruff validation passed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
