from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> int:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode


def main() -> int:
    print("Running final Ruff validation...")

    format_result = run(
        [
            "ruff",
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
            "ruff",
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