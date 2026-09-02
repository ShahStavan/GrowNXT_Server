import json
import sys

from validators.commands import validate_command
from validators.paths import validate_paths
from validators.security import validate_security


def main() -> int:
    payload = json.load(sys.stdin)

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {})

    if tool_name == "Bash":
        command = tool_input.get("command", "")

        validate_command(command)
        validate_security(command)

    elif tool_name in {"Write", "Edit"}:
        file_path = tool_input.get("file_path", "")

        validate_paths(file_path)
        validate_security(file_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())