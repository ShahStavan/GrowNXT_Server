---
name: code-style
description: The house Python style for GrowNXT Server and the refactor mandate that gets the existing modules there. Concise one-line docstrings, short trailing comments, flat modular functions. Load before writing or refactoring any module.
---

# Code Style

**The rule: code explains itself, comments explain the surprise, specs explain the design.**

Reference: [`ai-hedge-fund/src/agents/fundamentals.py`](https://github.com/ShahStavan/ai-hedge-fund/blob/main/src/agents/fundamentals.py).

Measured today: **16,456 lines, 21% of them prose** — 2,849 docstring lines and 543 comment
lines. The reference file runs about 5%. That gap is the mandate.

---

## Take from the reference

| Pattern | Example |
| :--- | :--- |
| One-line docstring | `"""Analyzes fundamental data and generates trading signals."""` |
| Short trailing comment | `if metrics["return_on_equity"] > 0.15:  # Strong ROE above 15%` |
| Numbered steps in a long function | `# 1. Profitability Analysis` … `# 2. Growth Analysis` |
| Flat functions over class hierarchies | `calculate_intrinsic_value(...)` is a plain function |
| Typed signature with defaults | `def f(x: float, rate: float = 0.05) -> float:` |

## Do not take from the reference

It is a prototype. These would be regressions here:

- `typing.Dict` / `Sequence` — use `dict`, `list`, `Sequence` from `collections.abc`.
- Untyped signatures (`def show_agent_reasoning(output, agent_name)`). **Every** parameter and
  return stays annotated.
- Missing docstrings (`merge_dicts` has none). One line, always.
- `##### Banner #####` comment blocks. Noise.
- No formatter config. Ruff at line-length 88 stays authoritative; the stop hook enforces it.
- Non-imperative docstrings are fine here — `"""Returns the quotient."""` stays. Ruff's `D401`
  would flag 275 of them for no readability gain, and the reference itself is non-imperative
  (`"""Analyzes fundamental data..."""`). `D401` is in `pyproject.toml`'s ignore list.
- 40-line functions doing five things. `fundamentals_agent` scores profitability, growth,
  health, ratios, runs a DCF, aggregates and builds a message. Split that.

---

## Rules

1. **Docstring: one line, imperative, ends in a period.** Add `Args:`/`Returns:` only when a
   signature is genuinely ambiguous — a units-bearing float, a tuple whose order is arbitrary, a
   sentinel return. Never restate the parameter names.
2. **No module docstring over 5 lines.** State what the module owns. Not why, not history.
3. **Comments are trailing and under ~60 characters.** A comment on its own line is for a
   numbered step or a real trap.
4. **No design rationale in code.** It goes to `.claude/specs/`. Cross-reference by path when it
   matters: `# see specs/vectorless-qualitative-rag.md §4.3`.
5. **No history in code.** "This was inverted once" belongs in a test name and a commit message.
6. **Functions under ~40 lines, one responsibility.** If you need numbered section comments to
   navigate it, it wants splitting.
7. **Skip `to_dict`/`from_dict` boilerplate.** `dataclasses.asdict` and a `**payload` constructor
   cover most of it. Hand-roll only for a schema that must survive a field rename.
8. **Drop `from __future__ import annotations`** where it buys nothing — Python 3.12 handles
   `X | None` and builtin generics natively. Keep it only for genuine forward references.
9. **Delete a comment that restates the line.** `# Initialize signals list` above
   `signals = []` is worse than nothing.

---

## Before / after

`reporting/fmt.py:144` — 5 lines of code, 21 of docstring:

```python
def pos_div(numerator: Any, denominator: Any) -> float | None:
    """Divides only when the denominator is strictly positive.

    This is the house rule for any ratio whose denominator can legitimately
    go negative: equity, EBITDA, EBIT, pre-tax profit, profit after tax,
    ... 15 more lines ...
    """
```

becomes:

```python
def pos_div(numerator: Any, denominator: Any) -> float | None:
    """Divides only when the denominator is strictly positive."""
    if not _is_number(numerator) or not _is_number(denominator):
        return None
    if float(denominator) <= 0:  # negative equity would flip ROE positive
        return None
    return float(numerator) / float(denominator)
```

The insight — negative profit over negative equity reads as strength on a typeset page — was
worth having. It is now nine words on the line that enforces it, and the full argument lives in
the reporting spec. **That is the move: keep the insight, lose the essay.**

---

## Where the deleted prose goes

| Content | Destination |
| :--- | :--- |
| Why this design won | `.claude/specs/*.md` |
| Why the old design lost | commit message |
| A trap that will bite the next reader | trailing comment, one line |
| A guarantee that must not regress | a test name in `scripts/verify_*.py` |
| What a function does | its one-line docstring |
| Nothing of the above | delete it |

---

## Refactor order

**Coverage first, not bloat first.** Prose ratio and test coverage are inversely related in this
repo, so ranking by bloat alone does the riskiest edits with no safety net. Full ordering,
per-phase gates and the modules that need tests written first are in
[`.claude/plans/code-style-refactor.md`](../../../.claude/plans/code-style-refactor.md).

The short version:

| Do now — well covered | Checks | Do after adding tests | Checks |
| :--- | :--- | :--- | :--- |
| `ingestion/documents/*` (55%, 45%, 41%, 27% prose) | 53 | `reporting/*` (25% and up, 7,761 lines) | 7 |
| `storage/gdrive.py` (24%) | 33 | `core/*` (49%, 34%) | **0** |
| `reporting/fmt.py` (43%) | covered directly | `scripts/cli.py` (43%), `api/*` | **0** |

Budget: **≤8% prose per module** (or 12 lines, whichever is larger — a 93-line module still
deserves a docstring) and no docstring over 5 lines without an `Args:` block earning it.

**The budget is enforced by `scripts/verify_style.py`, not by ruff.** No ruff rule caps docstring
length or measures a prose ratio, so without that check these numbers drift straight back — the
21% this tree carries was written under a Google-style mandate that also went unmeasured.

---

## Guardrails

A style refactor changes no behaviour. After **each** module:

```powershell
python scripts/parity_gate.py --quick
python scripts/parity_gate.py            # baseline 12/14, never lower
```

Never in the same commit as a behaviour change — a reviewer cannot separate them. One module
per commit, `style(<module>): ...`.

Do not touch while refactoring:

- Public signatures and `__all__`. Callers outside the module keep working.
- `EXTRACT_VERSION` and any version string folded into a cache key. Changing one invalidates
  every extraction on disk.
- The 20 self-checks in `reporting/selfcheck.py` and their names.
- Docstrings that are load-bearing for a caller: units, sign conventions, `None` semantics.

---

## Checklist

- [ ] One-line docstring on every public function, class and module
- [ ] `Args:`/`Returns:` only where the signature is genuinely ambiguous
- [ ] No comment restating its line; no history; no rationale
- [ ] Every parameter and return annotated; builtin generics, not `typing.Dict`
- [ ] No function over ~40 lines or with two responsibilities
- [ ] Rationale worth keeping moved to a spec, with the path referenced
- [ ] `ruff format` and `ruff check` clean
- [ ] Parity gate at 12/14 or better
