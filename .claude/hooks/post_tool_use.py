from __future__ import annotations

import json
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


def run_ruff(file_path: str) -> int:
    path = PROJECT_ROOT / file_path

    if not path.exists() or path.suffix != ".py":
        return 0

    ruff = _ruff_command()
    commands = [
        ruff + ["format", str(path)],
        ruff + ["check", str(path), "--fix"],
        ruff + ["check", str(path)],
    ]

    for command in commands:
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

        if result.returncode != 0:
            print(
                f"Ruff command failed: {' '.join(command)}",
                file=sys.stderr,
            )

            if result.stdout:
                print(result.stdout, file=sys.stderr)

            if result.stderr:
                print(result.stderr, file=sys.stderr)

            return 2

    print(f"Ruff validation passed: {file_path}")

    return 0


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(
            f"Invalid PostToolUse payload: {exc}",
            file=sys.stderr,
        )
        return 0

    tool_name = payload.get("tool_name", "")

    if tool_name not in {"Write", "Edit"}:
        return 0

    tool_input = payload.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    if not file_path.endswith(".py"):
        return 0

    return run_ruff(file_path)


if __name__ == "__main__":
    sys.exit(main())
