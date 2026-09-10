"""Style budget enforcement for the code-style refactor.

Ruff has no rule that caps a docstring's length or measures a prose ratio, so
the `code-style` skill's budget needs its own check or it drifts back.

"Prose" is narrative only: comments, plus docstring summary lines beyond the
one line per definition the skill mandates. `Args:`/`Returns:` blocks and that
mandated line are excluded -- counting either would penalise compliance.

Runs as a ratchet: `scripts/style_budget.json` records where each module
stands today, and a module that gets worse fails. `--strict` enforces the
final target instead, and `--update` lowers the ratchet after a refactor.

See `.claude/plans/code-style-refactor.md` Phase 0.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import cli  # noqa: E402
from scripts.checks import Report, banner  # noqa: E402

BUDGET_FILE = ROOT / "scripts" / "style_budget.json"

# TARGET_RATIO is the aspiration; MAX_RATIO is the gate. fmt.py obeys every
# rule the skill states and still lands at 10%. Plan C.1.
TARGET_RATIO = 0.08
MAX_RATIO = 0.15
TARGET_FLOOR = 12  # A small module still deserves a docstring.
MAX_SUMMARY = 5  # Docstring lines before the first Args:/Returns: block.
MAX_FUNCTION = 40  # Statements' line span, docstring excluded.

SKIP_DIRS = ("venv", "__pycache__", ".git", ".mypy_cache", ".ruff_cache", "output")

# Docstring section headers. Lines from here on are structured, not prose.
SECTION_RE = re.compile(
    r"^\s*(Args|Arguments|Returns|Yields|Raises|Attributes|Examples?|Note)s?:", re.M
)

# Design history belongs in a commit message, not in the source.
HISTORY_RE = re.compile(
    r"\b(was inverted|used to|previously|originally|no longer|formerly|"
    r"before this|in an earlier|the old |we used)\b",
    re.I,
)

LEGACY_GENERICS = ("Dict", "List", "Tuple", "Set", "FrozenSet", "Type", "Optional")

# Literals folded into a cache key or a documented count. A style refactor
# that changes one silently invalidates every extraction on disk. Plan §B.
FROZEN: dict[str, tuple[str, str]] = {
    "EXTRACT_VERSION": ("ingestion/documents/extract.py", "extract/docling-v1"),
    "FAST_TABLES_SUFFIX": ("ingestion/documents/extract.py", "+fast-tables"),
    "PAGE_FILTER_SUFFIX": ("ingestion/documents/sections.py", "+fin-pages"),
}

# 16 literal `name=` sites expand to the 20 runtime self-checks CLAUDE.md
# documents. Nothing else asserts this count -- verify_reporting.py checks
# that the checks pass, not that they all still exist.
SELFCHECK_SITES = 16


@dataclass
class ModuleStats:
    """Prose measurements and style findings for one module."""

    path: str
    lines: int = 0
    docstring: int = 0
    documented: int = 0
    comment: int = 0
    findings: dict[str, list[str]] = field(default_factory=dict)

    @property
    def prose(self) -> int:
        """Returns narrative prose only: comments plus over-long summaries.

        Two exclusions, both because counting them would penalise the style
        the skill mandates. The one-line docstring every definition owes: 13
        public accessors in storage.py owe 13 lines, already 8% of the file.
        And `Args:`/`Returns:` blocks, which the skill permits where a
        signature is ambiguous -- they are structured, not narrative.
        """
        return max(0, self.docstring - self.documented) + self.comment

    @property
    def ratio(self) -> float:
        """Returns prose as a fraction of total lines."""
        return self.prose / self.lines if self.lines else 0.0

    @property
    def allowance(self) -> int:
        """Prose lines the target permits."""
        return max(int(self.lines * TARGET_RATIO), TARGET_FLOOR)

    def add(self, kind: str, detail: str) -> None:
        """Records one finding of the given kind."""
        self.findings.setdefault(kind, []).append(detail)


def source_files() -> list[Path]:
    """Returns every project Python file, newest paths last."""
    out = [
        p
        for p in ROOT.glob("**/*.py")
        if not any(part in SKIP_DIRS for part in p.parts)
    ]
    return sorted(out)


def summary_length(docstring: str) -> int:
    """Returns the docstring's prose line count, excluding Args:-style blocks."""
    match = SECTION_RE.search(docstring)
    head = docstring[: match.start()] if match else docstring
    return len([ln for ln in head.splitlines() if ln.strip()])


def _public(name: str) -> bool:
    return not name.startswith("_")


def _check_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Returns a description of the first missing annotation, or None."""
    args = node.args
    every = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if args.vararg:
        every.append(args.vararg)
    if args.kwarg:
        every.append(args.kwarg)
    for arg in every:
        if arg.arg in ("self", "cls"):
            continue
        if arg.annotation is None:
            return f"{node.name}({arg.arg}) unannotated"
    if node.returns is None:
        return f"{node.name}() no return annotation"
    return None


def analyse(path: Path) -> ModuleStats | None:
    """Measures one module, returning None when it will not parse."""
    rel = path.relative_to(ROOT).as_posix()
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    lines = text.splitlines()
    stats = ModuleStats(path=rel, lines=len(lines))
    stats.comment = sum(1 for ln in lines if ln.strip().startswith("#"))

    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                # Summary lines only; an Args:/Returns: block is structured.
                stats.docstring += summary_length(doc)
                stats.documented += 1
                head = summary_length(doc)
                if head > MAX_SUMMARY:
                    name = getattr(node, "name", "<module>")
                    stats.add("docstring", f"{name} {head}L summary")

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _public(node.name):
                missing = _check_signature(node)
                if missing:
                    stats.add("annotations", missing)
            span = (node.end_lineno or node.lineno) - node.lineno
            doc = ast.get_docstring(node, clean=False)
            body = span - (len(doc.splitlines()) if doc else 0)
            if body > MAX_FUNCTION:
                stats.add("length", f"{node.name} {body} lines")

        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            for alias in node.names:
                if alias.name in LEGACY_GENERICS:
                    stats.add("generics", f"typing.{alias.name}")

    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#") and HISTORY_RE.search(stripped):
            stats.add("history", f"L{number}")

    return stats


def load_budget() -> dict:
    """Returns the recorded ratchet, or an empty one."""
    if not BUDGET_FILE.exists():
        return {}
    return json.loads(BUDGET_FILE.read_text(encoding="utf-8"))


def write_budget(modules: list[ModuleStats]) -> None:
    """Records current measurements as the new ratchet."""
    payload = {
        "_comment": "Ratchet for scripts/verify_style.py. Lower with --update; "
        "never raise by hand.",
        "target": {"ratio": TARGET_RATIO, "floor": TARGET_FLOOR},
        "modules": {
            m.path: {"lines": m.lines, "prose": m.prose, "docstring": m.docstring}
            for m in modules
            if m.lines
        },
        "findings": {
            kind: sum(len(m.findings.get(kind, [])) for m in modules)
            for kind in ("docstring", "annotations", "generics", "history", "length")
        },
    }
    BUDGET_FILE.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def check_ratchet(report: Report, modules: list[ModuleStats], budget: dict) -> None:
    """Fails any module carrying more prose than the ratchet allows."""
    report.section("Ratchet - prose must not grow")
    recorded = budget.get("modules", {})
    if not recorded:
        report.check("budget recorded", False, "run with --update to seed it")
        return

    worse = [
        f"{m.path} {m.prose} > {recorded[m.path]['prose']}"
        for m in modules
        if m.path in recorded and m.prose > recorded[m.path]["prose"]
    ]
    report.check(
        f"{len(modules)} modules at or below recorded prose",
        not worse,
        "; ".join(worse[:3]),
    )

    new = [m.path for m in modules if m.path not in recorded and m.ratio > TARGET_RATIO]
    report.check(
        "new modules meet the target",
        not new,
        f"{len(new)} over budget: {', '.join(new[:3])}",
    )


def check_findings(report: Report, modules: list[ModuleStats], budget: dict) -> None:
    """Fails any finding class with more occurrences than recorded."""
    report.section("Ratchet - findings must not grow")
    recorded = budget.get("findings", {})
    labels = {
        "docstring": f"docstring summaries over {MAX_SUMMARY} lines",
        "annotations": "public defs missing an annotation",
        "generics": "typing.Dict-style generics",
        "history": "history comments",
        "length": f"functions over {MAX_FUNCTION} lines",
    }
    for kind, label in labels.items():
        found = sum(len(m.findings.get(kind, [])) for m in modules)
        ceiling = recorded.get(kind)
        if ceiling is None:
            report.check(label, False, f"{found} found, none recorded")
            continue
        report.check(f"{label}: {found} (was {ceiling})", found <= ceiling)


def check_strict(report: Report, modules: list[ModuleStats]) -> None:
    """Fails any module over the hard ceiling, and reports the aspiration.

    A summary over `MAX_SUMMARY` is a rule violation; a ratio over
    `TARGET_RATIO` is usually just a well-documented module.
    """
    report.section(f"Ceiling - narrative prose <= {MAX_RATIO:.0%}")
    over = [m for m in modules if m.ratio > MAX_RATIO and m.prose > TARGET_FLOOR]
    for m in sorted(over, key=lambda m: m.ratio, reverse=True)[:12]:
        report.check(f"{m.path}", False, f"{m.prose} prose lines ({m.ratio:.0%})")
    report.check(
        f"{len(modules) - len(over)}/{len(modules)} modules under {MAX_RATIO:.0%}",
        not over,
    )

    report.section(f"Aspiration - prose <= {TARGET_RATIO:.0%} (reported only)")
    aspire = [m for m in modules if m.prose > m.allowance]
    report.check(
        f"{len(modules) - len(aspire)}/{len(modules)} modules at {TARGET_RATIO:.0%}",
        True,
        f"{len(aspire)} above it, none failing",
    )


def check_frozen(report: Report) -> None:
    """Fails if a cache-key literal or the self-check count moved."""
    report.section("Frozen literals (plan section B)")
    for name, (rel, expected) in FROZEN.items():
        text = (ROOT / rel).read_text(encoding="utf-8")
        report.check(f"{name} == {expected!r}", f'"{expected}"' in text, rel)

    sites = (ROOT / "reporting" / "selfcheck.py").read_text(encoding="utf-8")
    found = sites.count("name=")
    report.check(
        f"selfcheck.py has {SELFCHECK_SITES} name= sites",
        found == SELFCHECK_SITES,
        f"found {found}",
    )


def render_ranking(modules: list[ModuleStats]) -> str:
    """Returns the distance-to-target table, worst first."""
    rows = sorted(
        (m for m in modules if m.lines >= 40), key=lambda m: m.ratio, reverse=True
    )
    out = [
        "",
        f"  {'xs%':>5} {'lines':>6} {'doc':>5} {'cmt':>5} {'over':>6}  module",
    ]
    for m in rows[:15]:
        over = m.prose - m.allowance
        out.append(
            f"  {m.ratio:>5.0%} {m.lines:>6} {m.docstring:>5} {m.comment:>5} "
            f"{over:>+6}  {m.path}"
        )
    total_lines = sum(m.lines for m in modules)
    total_prose = sum(m.prose for m in modules)
    out.append(
        f"\n  tree: {total_prose}/{total_lines} prose = "
        f"{total_prose / total_lines:.0%} (target {TARGET_RATIO:.0%})"
    )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Runs the style checks and returns a process exit code."""
    parser = argparse.ArgumentParser(description="Style budget enforcement.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enforce the final target, not the ratchet.",
    )
    parser.add_argument(
        "--update", action="store_true", help="Record current state as the new ratchet."
    )
    parser.add_argument("--ranking", action="store_true", help="Print the table only.")
    args = parser.parse_args(argv)

    cli.console_utf8()
    modules = [s for s in (analyse(p) for p in source_files()) if s is not None]

    if args.ranking:
        print(render_ranking(modules))
        return 0

    if args.update:
        write_budget(modules)
        print(f"Recorded {len(modules)} modules to {BUDGET_FILE.name}.")
        print(render_ranking(modules))
        return 0

    banner("Style budget")
    report = Report()
    budget = load_budget()

    check_frozen(report)
    if args.strict:
        check_strict(report, modules)
    else:
        check_ratchet(report, modules, budget)
        check_findings(report, modules, budget)

    print(report.render())
    print(render_ranking(modules))
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
