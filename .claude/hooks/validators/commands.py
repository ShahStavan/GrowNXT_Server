import re
import sys


DANGEROUS_PATTERNS = [
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-[a-zA-Z]*f",
    r"\brm\s+-rf\s+/",
    r"\brm\s+-rf\s+\.",
    r"\bdel\s+/[sq]\b",
    r"\bformat\s+[a-zA-Z]:",
    r"\bdrop\s+database\b",
    r"\bdrop\s+table\b",
]


def validate_command(command: str) -> None:
    normalized = command.lower()

    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, normalized):
            print(
                f"BLOCKED: Potentially destructive command detected: {command}",
                file=sys.stderr,
            )
            raise SystemExit(2)