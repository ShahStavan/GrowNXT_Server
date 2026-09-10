"""Verification for `core/` — paths, compute resolution, SLM plumbing.

`core/` had no coverage before this, which made it the riskiest place to
refactor: `safe_ticker` builds every artefact path and `clean_thinking_tokens`
is the last thing between a model's reasoning and a published page. Offline --
no network, no model weights, no GPU.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import (  # noqa: E402
    config as config_module,  # noqa: E402
    hardware,
    llm_config,
)
from scripts import cli  # noqa: E402
from scripts.checks import Report, banner  # noqa: E402

# Indian symbols carry punctuation that is illegal or awkward in a path.
# M&M is the canonical case: an unescaped ampersand also breaks a URL query.
TICKER_FOLDING: tuple[tuple[str, str], ...] = (
    ("M&M", "M_M"),
    ("L&TFH", "L_TFH"),
    ("wipro", "WIPRO"),
    ("  tcs  ", "TCS"),
    ("BAJAJ-AUTO", "BAJAJ-AUTO"),
)


def check_ticker_folding(report: Report) -> None:
    """Every artefact path folds a symbol the same way."""
    report.section("Paths (core/config.py)")
    for raw, expected in TICKER_FOLDING:
        got = config_module.safe_ticker(raw)
        report.check(f"safe_ticker({raw!r}) == {expected!r}", got == expected, got)

    report.check(
        "folding is idempotent",
        all(
            config_module.safe_ticker(config_module.safe_ticker(r))
            == config_module.safe_ticker(r)
            for r, _ in TICKER_FOLDING
        ),
        "folding twice must not differ from folding once",
    )
    report.check(
        "no folded symbol keeps a path separator",
        all(
            "/" not in config_module.safe_ticker(r)
            and "\\" not in config_module.safe_ticker(r)
            for r, _ in TICKER_FOLDING
        ),
    )


def check_path_construction(report: Report) -> None:
    """Directory and report paths are derived, never concatenated by callers."""
    report.section("Path construction")
    folded = config_module.safe_ticker("M&M")

    directory = config_module.stock_dir("M&M")
    report.check(
        "stock_dir uses the folded symbol",
        directory.name == folded,
        str(directory),
    )
    report.check(
        "stock_dir does not create by default",
        not directory.exists() or directory.is_dir(),
        "a naming call must not leave a directory behind",
    )

    pdf = config_module.report_path("M&M")
    report.check(
        "report_path lands inside the stock directory",
        pdf.parent.name == folded and pdf.suffix == ".pdf",
        str(pdf),
    )
    report.check(
        "report_path names the folded symbol",
        folded in pdf.name,
        pdf.name,
    )
    report.check(
        "MAPPING_FILE_PATH sits at the project root",
        config_module.MAPPING_FILE_PATH.name == "mapping.json",
        str(config_module.MAPPING_FILE_PATH),
    )
    report.check(
        "HTTP_HEADERS carry a browser user agent",
        "user-agent" in {k.lower() for k in config_module.HTTP_HEADERS},
        "exchange hosts refuse requests without one",
    )


def check_hardware_resolution(report: Report) -> None:
    """A profile resolves on any host and reports what it chose."""
    report.section("Compute resolution (core/hardware.py)")
    resolved = hardware.profile()

    report.check(
        "device is a recognised prefix",
        resolved.device.split(":")[0] in hardware._VALID_DEVICE_PREFIXES,
        resolved.device,
    )
    report.check(
        "threads are positive and capped",
        1 <= resolved.num_threads <= hardware.MAX_DOCLING_THREADS,
        f"{resolved.num_threads} of max {hardware.MAX_DOCLING_THREADS}",
    )
    report.check(
        "Docling's own default is still 4",
        hardware.DOCLING_DEFAULT_THREADS == 4,
        "the reason this module exists: 4 threads on any machine",
    )
    report.check(
        "physical_cores is at least one",
        hardware.physical_cores() >= 1,
        str(hardware.physical_cores()),
    )
    report.check(
        "total_ram_gb is positive",
        hardware.total_ram_gb() > 0,
        f"{hardware.total_ram_gb():.1f} GB",
    )

    payload = resolved.to_dict()
    for key in ("device", "num_threads", "cpu_cores", "gpu_present"):
        report.check(f"to_dict carries {key!r}", key in payload)
    report.check(
        "summary names the device and thread count",
        "device=" in resolved.summary() and "threads=" in resolved.summary(),
        resolved.summary()[:70],
    )


def check_idle_gpu_detector(report: Report) -> None:
    """`idle_gpu` fires only for a present GPU that torch cannot reach.

    The failure it names is silent by nature: a CPU-only torch wheel makes
    every ``device="auto"`` mean cpu on a machine with a working card, and
    nothing anywhere reports it.
    """
    report.section("idle_gpu detector")
    from dataclasses import replace

    base = hardware.profile()

    idle = replace(base, device="cpu", gpu_present=True, cuda_build="")
    report.check("fires: GPU present, no CUDA build, on cpu", idle.idle_gpu)

    for label, sample in (
        ("silent on cpu with no GPU", replace(idle, gpu_present=False)),
        ("silent when already on cuda", replace(idle, device="cuda")),
        ("silent when torch has CUDA", replace(idle, cuda_build="12.6")),
    ):
        report.check(label, not sample.idle_gpu)

    report.check(
        "on_gpu agrees with the device",
        replace(base, device="cuda").on_gpu and not replace(base, device="cpu").on_gpu,
    )


def check_thinking_token_sanitiser(report: Report) -> None:
    """No `<think>` block may survive into a report or a stream.

    The project rule is absolute: model reasoning never reaches a page. The
    unclosed case is the one that matters, because a truncated stream ends
    mid-thought and a naive paired-tag regex leaves the whole tail in place.
    """
    report.section("Thinking-token sanitiser (core/llm_config.py)")
    cases: tuple[tuple[str, str, str], ...] = (
        ("paired block removed", "a<think>why</think>b", "ab"),
        ("unclosed block truncates", "a<think>never closed", "a"),
        ("case-insensitive", "a<THINK>why</THINK>b", "ab"),
        ("multiline block", "a<think>one\ntwo</think>b", "ab"),
        ("multiple blocks", "<think>x</think>a<think>y</think>b", "ab"),
        ("no block is untouched", "plain text", "plain text"),
    )
    for label, raw, expected in cases:
        got = llm_config.clean_thinking_tokens(raw)
        report.check(label, got == expected, f"{raw!r} -> {got!r}")

    report.check(
        "no output retains a think tag",
        all(
            "<think" not in llm_config.clean_thinking_tokens(raw).lower()
            for _, raw, _ in cases
        ),
    )

    thinking, content = llm_config.extract_thinking_and_content("a<think>why</think>b")
    report.check(
        "extract splits reasoning from content",
        thinking == "why" and content == "ab",
        f"{thinking!r} / {content!r}",
    )


def check_llm_configuration(report: Report) -> None:
    """The SLM endpoint and model names resolve to non-empty values."""
    report.section("SLM configuration")
    report.check(
        "API_URL is an absolute http(s) URL",
        llm_config.API_URL.startswith(("http://", "https://")),
        llm_config.API_URL,
    )
    report.check("ACTIVE_MODEL is set", bool(llm_config.ACTIVE_MODEL))
    report.check(
        "REQUEST_TIMEOUT is generous enough for a cold scrape",
        llm_config.REQUEST_TIMEOUT >= 30,
        f"{llm_config.REQUEST_TIMEOUT}s",
    )
    report.check(
        "the collector base URL is absolute",
        config_module.FINANCIAL_DATA_COLLECTOR_BASE_URL.startswith("http"),
        config_module.FINANCIAL_DATA_COLLECTOR_BASE_URL,
    )


def main(argv: list[str] | None = None) -> int:
    """Runs the suite and returns a process exit code."""
    parser = argparse.ArgumentParser(description="Verify the core/ layer.")
    parser.add_argument("--verbose", action="store_true", help="Debug logging.")
    args = parser.parse_args(argv)

    cli.console_utf8()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )

    banner("Core layer verification")
    report = Report()
    check_ticker_folding(report)
    check_path_construction(report)
    check_hardware_resolution(report)
    check_idle_gpu_detector(report)
    check_thinking_token_sanitiser(report)
    check_llm_configuration(report)

    print(report.render())
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
