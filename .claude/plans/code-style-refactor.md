# Plan: Refactor the Codebase to the `code-style` Skill

- **Status**: Ready for implementation, pending the 3 decisions in §C
- **Skill**: [`.agents/skills/code-style/SKILL.md`](../../.agents/skills/code-style/SKILL.md)
- **Author**: Claude, for @jenish.gajera
- **Date**: 2026-09-10
- **Branch**: `feat/vectorless-qualitative-rag` (continues; no new branch — style commits interleave
  with the qualitative rebuild)
- **Headline**: the skill ranked modules by prose ratio. **Prose and test coverage are
  anti-correlated here** (§A/F1), so shipping in skill order does the riskiest edits with no
  safety net. This plan reorders by coverage. It also fixes the skill's fatal gap: **no ruff rule
  exists for docstring length** (§A/F2), so without a custom check the 21% → 8% gain regresses on
  the next commit.

---

## A. Verification against the code

Every figure in the skill was re-derived from the AST. These verified true:

| Skill claim | Verified |
| :--- | :--- |
| 16,456 lines; 2,849 docstring + 543 comment = 21% prose | ✅ AST walk over all non-venv `*.py` |
| `ingestion/documents/__init__.py` 55% prose, 49-line module docstring | ✅ 105 lines, 54 doc |
| `core/config.py` 49%; `report_path` docstring 13 lines | ✅ 13L docstring, **2 statements** |
| `ingestion/documents/storage.py` 45%, 21-line module docstring | ✅ |
| `reporting/fmt.py` 43%; `pos_div` 21-line docstring, 5 lines of code | ✅ 21L docstring, 3 statements |
| `scripts/cli.py` 43%; `setup` docstring 8 lines | ✅ 8L docstring, 2 statements |
| `ingestion/documents/sections.py` 41%, 31-line module docstring | ✅ |
| `core/hardware.py` 34%, 36-line module docstring | ✅ |
| `reporting/composites.py` 363 doc lines, 44-line module docstring | ✅ largest absolute |
| Ruff line-length 88, `E501` ignored in lint | ✅ `pyproject.toml:54,74` |

### F1 — 🔴 The skill's ordering is unsafe. Bloat and coverage are inversely related.

Behavioural checks per module group, measured:

| Area | Lines | Checks | Density | Holds skill targets |
| :--- | ---: | ---: | :--- | :--- |
| `ingestion/documents/` | 2,590 | 53 | **1 per 49 lines** | #1, #3, #6, #8 |
| `storage/` | 581 | 33 | **1 per 18 lines** | — |
| `reporting/` | 7,761 | 7 | 1 per 1,109 lines | #4, #7, #9 |
| `core/` | 863 | **0** | none | **#2, #10** |
| `ingestion/catalog.py`, `fetcher.py` | 723 | **0** | none | — |
| `api/` + `app.py` | 425 | **0** | none | — |
| `scripts/cli.py`, `checks.py` | 214 | **0** | none | **#5** |

The skill's #2 (`core/config.py`) and #5 (`scripts/cli.py`) have **zero** coverage. Its #7
(`composites.py`, 1,646 lines) sits behind 7 checks shared across all 7,761 lines of
`reporting/`. Meanwhile `ingestion/documents/` — the best-tested code in the repo at 1 check per
49 lines — holds four of the ten targets including the worst offender.

**Therefore: reorder coverage-first.** `ingestion/documents/` and `storage/` are free wins.
`reporting/` needs checks added before its big files are touched. `core/`, `api/` and
`scripts/` helpers are refactored only after they have a check apiece.

`reporting/fmt.py` is the one happy exception: `check_division_helpers`
(`verify_reporting.py:212`) exercises `pos_div` directly, so the skill's emblematic example is
safe to do immediately.

### F2 — 🔴 The skill's core rule is not machine-enforceable. Ruff has no docstring-length rule.

Searched the full rule set: the only length-adjacent docstring settings are
`lint.pydoclint.ignore-one-line-docstrings` and the formatter's line-length. **Nothing caps a
docstring at N lines, and nothing measures a prose ratio.**

So "≤8% prose, no docstring over 5 lines" is a convention with no enforcement. On current
evidence this repo drifts back: the 21% it carries today was written *under* a Google-style
mandate that also went unenforced.

**Fix**: `scripts/verify_style.py`, wired into `parity_gate.py`. The measurement code already
exists — it produced every table in this plan. It needs a per-file budget and an allowlist for
genuine `Args:` blocks.

### F3 — 🟠 Adopt a curated `D` subset, and explicitly reject `D401`

`ruff check --select D` yields **390 errors, 84 auto-fixable**. Breakdown:

| Rule | Count | Verdict |
| :--- | ---: | :--- |
| `D401` non-imperative-mood | **275** | **Reject.** Wants `"""Report whether..."""` over `"""Reports whether..."""`. 275 first-word edits, zero readability gain — and the reference file itself is non-imperative (`"""Analyzes fundamental data..."""`). Adopting it would contradict the style we are copying. |
| `D413` missing-blank-line-after-last-section | 81 | Adopt — auto-fixable, cosmetic consistency. |
| `D102`/`D103`/`D100`/`D107`/`D105` undocumented | 32 | Adopt. Aligns with "one line, always". |
| `D202`/`D203`/`D211`/`D212` blank lines | 5 | Adopt the compatible ones; ruff warns `D203`/`D211` and `D212`/`D213` conflict — pick `D211` and `D212`. |

Net: `select` gains `D`, `ignore` gains `D401`. About 115 real findings, most auto-fixable.

### F4 — 🟠 `pyproject.toml` carries dead config from my own cleanup

`pyproject.toml:113-125` holds per-file-ignores for `scripts/ingest_documents.py` and
`scripts/verify_ingestion.py` — **both deleted in `949240d`** — plus a 7-line comment describing
the pre-Docling architecture (`DocumentRegistry`, `IngestionPipeline`, `ingestion.layout`). I
missed it in `22edd4b`. Remove in Phase 0.

### F5 — 🟢 `E501` being ignored is correct and should stay

Lint ignores `E501` while the formatter enforces `line-length = 88`. That means an over-long
comment is *wrapped*, never rejected — which is what we want while shortening prose. Leave it.

### F6 — 🟡 Some long docstrings have nowhere to go

The skill's relocation table sends design rationale to `.claude/specs/`. But `specs/` holds only
the three Nifty 50 / vectorless documents. **There is no spec for `reporting/`**, and that is
where the densest rationale lives — `composites.py` (44-line module docstring), `selfcheck.py`
(32), `snapshot.py`, `charts.py` (21), `fmt.py`.

Relocation with no destination becomes deletion. Phase 3 therefore opens by *creating*
`.claude/specs/reporting-quantitative-engine.md` as the receiving document. Same for
`core/hardware.py`'s 36-line docstring, whose content is the `idle_gpu` reasoning — that belongs
in the vectorless spec, which already discusses it (§5).

---

## B. Scope

**In scope**: docstrings, comments, function decomposition, `typing.Dict` → builtins, dead
`from __future__ import annotations`, `to_dict`/`from_dict` boilerplate where `asdict` suffices.

**Out of scope**: behaviour, public signatures, `__all__`, file boundaries, dependencies, and any
version string folded into a cache key.

**Frozen, per the skill's guardrails** — verified present and load-bearing:

| Frozen thing | Where | Why |
| :--- | :--- | :--- |
| `EXTRACT_VERSION`, `extract_version()` | `documents/extract.py:64,128` | folded into `_cache_key`; a change invalidates every extraction on disk |
| `PAGE_FILTER_SUFFIX`, `FAST_TABLES_SUFFIX` | `sections.py:53`, `extract.py:81` | same cache key |
| The 20 self-check names | `reporting/selfcheck.py` | asserted by `check_cached_reports_verify` |
| `pos_div` / `pos_ratio` sign semantics | `reporting/fmt.py` | the one docstring whose *content* is a contract |
| `MAPPING_FILE_PATH`, `safe_ticker` | `core/config.py:27` | path construction across the tree |

---

## C. Decisions needed

### C.1 — Prose budget: flat 8%, or per-module?

A flat 8% punishes small files: `core/config.py` at 93 lines allows ~7 prose lines total across a
module docstring and 6 functions. **Recommendation**: budget `max(8% of lines, 12 lines)`, so
small modules keep a module docstring plus one-liners. Tune once `verify_style.py` reports real
numbers.

### C.2 — Does `reporting/typst_doc.py` (2,168 lines) get split?

It is the largest file and only 12% prose, so the skill's prose rule barely touches it. Its real
problem is a 27-line docstring on `build_document` and the file's size. Splitting it is a
structural change, not a style one. **Recommendation**: out of scope here; shorten the docstring,
leave the structure, and raise splitting separately.

### C.3 — Interleave with the qualitative rebuild, or finish first?

Refactoring `ingestion/documents/` (Phase 1) touches exactly the modules the vectorless
pre-processing work builds on. **Recommendation**: do Phase 0 and Phase 1 now, before
pre-processing starts writing against those modules — cheaper than refactoring around new code.
Phases 3-5 can wait.

---

## D. Acceptance criteria per module

1. `verify_style.py` reports the module within budget.
2. No docstring over 5 lines without an `Args:`/`Returns:` block earning it.
3. Every parameter and return annotated; builtin generics.
4. No function over ~40 lines or with two responsibilities.
5. Relocated rationale is in a spec, and the module references it by path.
6. Parity gate at its current baseline or better.
7. `git diff --stat` shows **no change to executable lines** unless the commit says otherwise.

Criterion 7 is the important one: a style commit whose diff touches logic is a bug in the commit,
not a bonus.

---

## E. Phases

### Phase 0 — Enforcement first (half day)

Nothing is refactored until the ratchet exists, or Phase 1's gains leak back.

- `scripts/verify_style.py` — per-module prose ratio, docstring-length findings, budget from C.1,
  built on `scripts/checks.py`. Promote the AST measurement used in this plan.
- Wire it into `parity_gate.py` `SUITES`, and record the new baseline.
- `pyproject.toml`: add `D`, ignore `D401` (F3); delete the dead per-file-ignores (F4).
- `ruff check --select D --fix` for the 84 auto-fixable.

*Gate*: parity gate green at its new baseline; `verify_style.py` reports today's 21% as failing,
with a per-module ranking that matches this plan's tables.

### Phase 1 — `ingestion/documents/` (1 day) — best coverage, worst bloat

Targets #1, #3, #6, #8 behind 53 checks. `__init__.py` 49→5, `storage.py` 21→5,
`sections.py` 31→5, `extract.py`'s 30-line `run` docstring, `content.py` 24-line.

`sections.py` needs care: its docstring holds the auditor's-certificate finding (page 191
"CERTIFICATE" vs "REPORT"), which is a genuine trap. It becomes one trailing comment on the
`ANCHORS` entry plus a line in the vectorless spec, which already records it.

*Gate*: `verify_documents.py` 53/53 unchanged; `verify_style.py` passes all five modules.

### Phase 2 — `storage/` and `reporting/fmt.py` (half day)

`storage/gdrive.py` 26-line module docstring behind 33 checks. `fmt.py`'s `pos_div` is the
skill's worked example and is covered by `check_division_helpers`.

### Phase 3 — `reporting/` (2 days) — **checks before edits**

Open by creating `.claude/specs/reporting-quantitative-engine.md` (F6) as the destination.
Then, in this order:

1. Raise `verify_reporting.py` from 7 checks to ~25, targeting `composites.py`, `analytics.py`,
   `charts.py` and `tokens.py` directly. **This is most of the phase, and it is the point** —
   1,646 lines of composites behind a shared 7 checks is not refactorable.
2. Only then shorten `composites.py` (363 doc lines), `selfcheck.py`, `snapshot.py`, `charts.py`,
   and `typst_doc.py`'s `build_document` docstring per C.2.

### Phase 4 — `core/` (1 day) — zero coverage today

`config.py` (#2), `hardware.py` (#10), `llm_config.py`. Add `scripts/verify_core.py` first —
`safe_ticker` folding of `M&M`, `report_path`, and `hardware.profile()`/`resolve_workers()`
resolution are all testable and currently untested. `hardware.py`'s rationale goes to the
vectorless spec §5, which already covers `idle_gpu`.

### Phase 5 — `api/`, `app.py`, `scripts/` helpers (half day)

Lowest prose ratios (6-10%), so mostly `D`-rule cleanup. `scripts/cli.py` (#5, 43%) is small and
its `setup` docstring is 8 lines for 2 statements.

**Total: ~5.5 days**, of which Phase 3's test-writing is ~1.5. Phases 0-2 are ~2 days and cover
six of the skill's top ten.

---

## F. Verification plan

`scripts/verify_style.py` checks:

| Check | Guards |
| :--- | :--- |
| `check_prose_budget` | every module within C.1's budget |
| `check_docstring_length` | none over 5 lines without an `Args:` block |
| `check_no_history_comments` | flags `was inverted`, `used to`, `previously`, `originally` |
| `check_annotations_complete` | every public def fully annotated |
| `check_builtin_generics` | no `typing.Dict`/`List`/`Tuple` |
| `check_frozen_strings` | §B's version constants unchanged — a literal allowlist |
| `check_function_length` | flags over 40 lines, ignoring docstring |

`check_frozen_strings` is the one that matters most: it makes the cache-invalidation footgun in
§B mechanical rather than remembered.

---

## G. Risks

| Risk | Severity | Mitigation |
| :--- | :--- | :--- |
| A refactor silently changes behaviour in `reporting/` | **High** | Phase 3 writes checks first, and criterion 7 forbids executable-line changes in a style commit |
| Relocated rationale is dropped rather than moved | **High** | F6 creates the destination spec *before* the phase; `check_no_history_comments` catches the leftovers, not the losses — so review each relocation in the PR |
| The auditor's-certificate and `should_index_document` findings are lost | Medium | Both already live in the vectorless spec and CLAUDE.md gotchas; Phase 1 verifies the reference survives before deleting the docstring |
| `D401` gets adopted by reflex from a "select D" suggestion | Medium | F3 records the rejection and the reason in `pyproject.toml` as a comment |
| 5.5 days of no feature work | Medium | C.3 — do Phases 0-1 now because pre-processing builds on those modules; defer 3-5 |
| Prose budget is wrong and churns files twice | Low | Phase 0 ships the measurement before any edit, so C.1 is tuned on data |

---

## H. Checklist

- [ ] Phase 0 — `verify_style.py`, `D` minus `D401`, dead per-file-ignores removed, baseline recorded
- [ ] Phase 1 — `ingestion/documents/` five modules; 53/53 checks hold
- [ ] Phase 2 — `storage/gdrive.py`, `reporting/fmt.py`
- [ ] Phase 3 — reporting spec created; `verify_reporting.py` 7 → ~25 checks; then the big files
- [ ] Phase 4 — `verify_core.py` added; `core/` three modules
- [ ] Phase 5 — `api/`, `app.py`, `scripts/cli.py`
- [ ] Tree-wide prose ratio ≤8%, from 21%
- [ ] Skill updated: refactor order corrected to coverage-first, `verify_style.py` referenced
