import re
import sys


SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{20,}",
    r"-----BEGIN .* PRIVATE KEY-----",
    r"(?i)(api[_-]?key|client[_-]?secret)\s*[:=]\s*['\"][^'\"]+",
    r"(?i)authorization:\s*bearer\s+[A-Za-z0-9._-]+",
]


def validate_security(value: str) -> None:
    for pattern in SECRET_PATTERNS:
        if re.search(pattern, value):
            print(
                "BLOCKED: Possible credential or secret detected.",
                file=sys.stderr,
            )
            raise SystemExit(2)