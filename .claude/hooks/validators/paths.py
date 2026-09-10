import sys
from pathlib import Path

PROTECTED_FILES = {
    ".env",
    ".env.production",
    "credentials.json",
}


PROTECTED_DIRECTORIES = {
    ".git",
}


def validate_paths(file_path: str) -> None:
    path = Path(file_path)

    if path.name in PROTECTED_FILES:
        print(
            f"BLOCKED: Direct modification of protected file: {file_path}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if any(part in PROTECTED_DIRECTORIES for part in path.parts):
        print(
            f"BLOCKED: Modification inside protected directory: {file_path}",
            file=sys.stderr,
        )
        raise SystemExit(2)
