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

By measured prose ratio and absolute docstring lines. Highest leverage first.

| # | Module | Lines | Doc | Prose | Target |
| :--- | :--- | ---: | ---: | ---: | :--- |
| 1 | `ingestion/documents/__init__.py` | 105 | 54 | **55%** | 49-line module docstring → 5 |
| 2 | `core/config.py` | 93 | 40 | **49%** | 13-line `report_path` docstring → 1 |
| 3 | `ingestion/documents/storage.py` | 203 | 80 | **45%** | 21-line module docstring → 5 |
| 4 | `reporting/fmt.py` | 216 | 92 | **43%** | the `pos_div` case above |
| 5 | `scripts/cli.py` | 54 | 20 | **43%** | 8-line `setup` docstring → 1 |
| 6 | `ingestion/documents/sections.py` | 167 | 52 | **41%** | 31-line module docstring → 5 |
| 7 | `reporting/composites.py` | 1,646 | 363 | 25% | largest absolute; 44-line module docstring |
| 8 | `ingestion/documents/extract.py` | 1,056 | 217 | 27% | 30-line `run` docstring; split the function |
| 9 | `reporting/typst_doc.py` | 2,168 | 210 | 12% | biggest file; 27-line `build_document` |
| 10 | `core/hardware.py` | 455 | 129 | 34% | 36-line module docstring |

Aim for **≤8% prose per module** and no docstring over 5 lines outside a genuine `Args:` block.

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
