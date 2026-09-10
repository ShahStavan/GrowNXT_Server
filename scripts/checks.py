"""Shared harness for the verification scripts.

Three suites -- reporting, ingestion and Drive delivery -- each accumulated
pass/fail, padded a listing and printed a banner in its own way, one of them
through module-level counters. The mechanics were identical, so they live here
and each suite is left holding only its own assertions.

A check records rather than raises. A verification run should report every
finding in one pass: stopping at the first failure hides how much else broke,
which is exactly what you need to know when deciding whether a change is
salvageable.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

RULE: str = "=" * 72
DETAIL_LIMIT: int = 70


class Failure(Exception):
    """Raised inside a check to report that it did not hold."""


def require(condition: bool, message: str) -> None:
    """Raises `Failure` with `message` unless `condition` holds."""
    if not condition:
        raise Failure(message)


@dataclass
class Check:
    """One recorded check, or a section heading when `passed` is None.

    Headings share the list so that rendering keeps them interleaved with the
    checks they introduce, rather than needing a second structure.
    """

    name: str
    passed: bool | None
    detail: str = ""

    @property
    def is_heading(self) -> bool:
        """Whether this entry is a section heading rather than a check."""
        return self.passed is None


@dataclass
class Report:
    """Accumulates checks and renders them as one listing."""

    entries: list[Check] = field(default_factory=list)

    # -- recording ---------------------------------------------------------

    def section(self, title: str) -> None:
        """Opens a titled group in the listing."""
        self.entries.append(Check(title, None))

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        """Records one check and returns its outcome.

        The outcome is returned so a caller can skip dependent checks that
        would only produce noise once their precondition has failed.
        """
        held = bool(condition)
        self.entries.append(Check(name, held, detail))
        return held

    def raises(
        self,
        name: str,
        kind: type[BaseException],
        call: Callable[[], object],
        detail: str = "",
    ) -> bool:
        """Records that `call` raises `kind`, and nothing else.

        A wrong exception type is a distinct failure from no exception at all,
        and both are reported as what actually happened.
        """
        try:
            call()
        except kind as exc:
            return self.check(name, True, detail or str(exc)[:DETAIL_LIMIT])
        except Exception as exc:  # noqa: BLE001 - the wrong type is the finding
            return self.check(
                name, False, f"raised {type(exc).__name__} instead: {exc}"
            )
        return self.check(name, False, f"did not raise {kind.__name__}")

    def run(self, name: str, call: Callable[[], str]) -> bool:
        """Records a coarse check that returns its own detail, or raises.

        For suites whose checks assert internally with `require` and describe
        what they observed on the way through.
        """
        try:
            return self.check(name, True, call())
        except Failure as exc:
            return self.check(name, False, str(exc))
        except Exception as exc:  # noqa: BLE001 - report and keep going
            return self.check(name, False, f"{type(exc).__name__}: {exc}")

    def extend(self, other: "Report") -> None:
        """Absorbs another report's entries, preserving order."""
        self.entries.extend(other.entries)

    # -- reporting ---------------------------------------------------------

    @property
    def results(self) -> list[Check]:
        """Every recorded check, headings excluded."""
        return [e for e in self.entries if not e.is_heading]

    @property
    def failures(self) -> list[Check]:
        """The checks that did not hold."""
        return [e for e in self.results if not e.passed]

    def render(self) -> str:
        """Returns the listing, one line per check, headings interleaved."""
        width = max([len(e.name) for e in self.results] + [4])
        lines = []
        for entry in self.entries:
            if entry.is_heading:
                lines.extend(["", entry.name])
            else:
                lines.append(
                    f"  [{'PASS' if entry.passed else 'FAIL'}] {entry.name:<{width}}  {entry.detail}"
                )
        results, failed = self.results, self.failures
        lines.extend(
            [
                "",
                f"  {len(results) - len(failed)} of {len(results)} checks passed.",
            ]
        )
        return "\n".join(lines)

    def finish(self) -> int:
        """Prints the closing banner and returns a process exit code."""
        print("\n" + RULE)
        if not self.failures:
            print(f"All {len(self.results)} checks passed.")
            return 0
        print(
            f"FAILED: {len(self.failures)} of {len(self.results)} checks did not hold."
        )
        for entry in self.failures:
            print(f"  - {entry.name}: {entry.detail}")
        return 1


def banner(title: str) -> None:
    """Prints a suite's opening banner."""
    print(RULE)
    print(title)
    print(RULE)
