from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_ruff(file_path: str) -> int:
    path = PROJECT_ROOT / file_path

    if not path.exists() or path.suffix != ".py":
        return 0

    commands = [
        ["ruff", "format", str(path)],
        ["ruff", "check", str(path), "--fix"],
        ["ruff", "check", str(path)],
    ]

    for command in commands:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

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