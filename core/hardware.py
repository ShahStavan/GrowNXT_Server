"""Resolves the compute this machine can actually give the ingestion pipeline.

`profile()` reports what the host has; `configure()` applies it and returns
what it applied. Exists because Docling pins 4 threads on any machine and
resolves ``device="auto"`` to CPU on a CPU-only torch wheel, silently:
`.claude/specs/vectorless-qualitative-rag.md` section 11.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Final

logger = logging.getLogger(__name__)

# Docling's own default, restated so the log can say what was overridden.
DOCLING_DEFAULT_THREADS: Final[int] = 4

# Embedding batch by VRAM. Snowflake Arctic m-v1.5 is 109M parameters at 512
# tokens, so a batch of 32 in fp16 is well under a gigabyte of activations --
# the headroom is for Docling's layout and table models, which sit on the same
# card for the whole run and are the reason these numbers are not larger.
_VRAM_BATCH: Final[tuple[tuple[float, int], ...]] = (
    (6.0, 32),
    (12.0, 64),
    (24.0, 128),
)
_BATCH_ABOVE_24GB: Final[int] = 256

# CPU embedding is memory-bandwidth-bound rather than parallel, and a large
# batch on a machine with little RAM buys nothing while risking a swap storm.
_CPU_BATCH: Final[int] = 64
_CPU_BATCH_LOW_RAM: Final[int] = 16
_LOW_RAM_GB: Final[float] = 12.0

# Above this, more Docling threads stop helping: the pipeline serialises on the
# PDF parser between model passes, and every thread holds page-sized buffers.
MAX_DOCLING_THREADS: Final[int] = 16

_VALID_DEVICE_PREFIXES: Final[tuple[str, ...]] = ("cuda", "cpu", "mps", "xpu")


def _env(name: str) -> str:
    """Returns a stripped environment variable, or the empty string."""
    return (os.getenv(name) or "").strip()


def _env_int(name: str) -> int | None:
    """Returns an environment variable parsed as a positive int, or None."""
    raw = _env(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; ignoring.", name, raw)
        return None
    return value if value > 0 else None


def _env_flag(name: str, default: bool) -> bool:
    """Returns an environment variable read as a boolean."""
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def physical_cores() -> int:
    """Returns the machine's physical core count, best effort.

    Docling recommends physical cores rather than logical ones: its model
    passes are already vectorised, so a second thread on the same core competes
    for the same execution units instead of adding throughput.
    """
    try:
        import psutil

        cores = psutil.cpu_count(logical=False)
        if cores:
            return int(cores)
    except Exception:  # noqa: BLE001 - psutil is optional at runtime
        pass
    logical = os.cpu_count() or 1
    # No physical count available: assume SMT, which is right on every laptop
    # this runs on and merely conservative where it is not.
    return max(1, logical // 2) if logical > 1 else 1


def total_ram_gb() -> float:
    """Returns installed RAM in GB, or 0.0 when it cannot be determined."""
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:  # noqa: BLE001 - psutil is optional at runtime
        return 0.0


def nvidia_gpu_present() -> bool:
    """Whether the machine appears to have an NVIDIA GPU, independent of torch.

    Asked separately from `detect_accelerator` so the two answers can disagree:
    a card here and no CUDA there is the CPU-wheel trap, while agreement on
    "no GPU" is just a CPU machine and warrants no warning at all.

    Presence of the driver's own tool is the cheap signal -- it ships with
    every NVIDIA driver on Windows and Linux -- and it is only consulted, never
    executed.
    """
    return shutil.which("nvidia-smi") is not None


def detect_accelerator() -> tuple[str, str, float]:
    """Returns the best available ``(device, name, vram_gb)``.

    Falls back to ``("cpu", "", 0.0)`` when torch is missing or was installed
    without a CUDA runtime -- the case this module exists to make visible,
    since a CPU-only wheel makes Docling's ``device="auto"`` mean ``cpu``
    on a machine with a perfectly good GPU in it.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch is an ML extra
        return "cpu", "", 0.0

    try:
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            return "cuda", str(props.name), round(props.total_memory / (1024**3), 1)
    except Exception as exc:  # noqa: BLE001 - a broken driver is not a crash
        logger.debug("CUDA probe failed: %s", exc)

    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    try:
        if mps is not None and mps.is_available():
            return "mps", "Apple Silicon", 0.0
    except Exception:  # noqa: BLE001
        pass

    xpu = getattr(torch, "xpu", None)
    try:
        if xpu is not None and xpu.is_available():
            return "xpu", "Intel XPU", 0.0
    except Exception:  # noqa: BLE001
        pass

    return "cpu", "", 0.0


def _embed_batch_size(device: str, vram_gb: float, ram_gb: float) -> int:
    """Returns texts per encode call for the resolved device."""
    if device.startswith("cuda") and vram_gb > 0:
        for ceiling, size in _VRAM_BATCH:
            if vram_gb < ceiling:
                return size
        return _BATCH_ABOVE_24GB
    if device.startswith(("mps", "xpu", "cuda")):
        return _CPU_BATCH
    if ram_gb and ram_gb < _LOW_RAM_GB:
        return _CPU_BATCH_LOW_RAM
    return _CPU_BATCH


@dataclass(frozen=True)
class HardwareProfile:
    """What the machine has, and what each compute stage should be given.

    Attributes:
        device: Torch/Docling device string -- ``cuda``, ``cuda:1``, ``cpu``,
            ``mps`` or ``xpu``. Never ``auto``: this is the resolved answer.
        gpu_name: Accelerator model name, empty on CPU.
        vram_gb: Accelerator memory, 0.0 when not applicable.
        cpu_cores: Physical cores.
        ram_gb: Installed RAM.
        num_threads: Threads given to Docling and to torch's intra-op pool.
        embed_batch_size: Texts per `SentenceTransformer.encode` call.
        embed_fp16: Whether to run the embedding model in half precision.
        torch_build: Installed torch version string, empty when absent.
        cuda_build: CUDA version torch was compiled against, empty on a CPU
            wheel. A CUDA-capable card plus an empty ``cuda_build`` is exactly
            the "GPU sitting idle" case.
        gpu_present: Whether an NVIDIA driver was found on the machine,
            established without torch's help.
        overrides: Names of the settings that came from the environment or a
            caller rather than from detection.

    """

    device: str = "cpu"
    gpu_name: str = ""
    vram_gb: float = 0.0
    cpu_cores: int = 1
    ram_gb: float = 0.0
    num_threads: int = DOCLING_DEFAULT_THREADS
    embed_batch_size: int = _CPU_BATCH
    embed_fp16: bool = False
    torch_build: str = ""
    cuda_build: str = ""
    gpu_present: bool = False
    overrides: tuple[str, ...] = ()

    @property
    def on_gpu(self) -> bool:
        """Whether model inference will run on an accelerator."""
        return not self.device.startswith("cpu")

    @property
    def idle_gpu(self) -> bool:
        """Whether a CUDA-capable card is present but torch cannot reach it.

        True only for the specific, silent, and very expensive case: torch is
        installed from the CPU wheel index on a machine that has an NVIDIA GPU.
        Callers surface this as a warning rather than acting on it.
        """
        return self.gpu_present and not self.cuda_build and not self.on_gpu

    def to_dict(self) -> dict[str, Any]:
        """Returns the profile as JSON-serialisable data for logs and manifests."""
        data = asdict(self)
        data["overrides"] = list(self.overrides)
        return data

    def summary(self) -> str:
        """Returns a single operator-facing line describing the profile."""
        where = (
            f"{self.device} ({self.gpu_name}, {self.vram_gb:g} GB)"
            if self.on_gpu and self.gpu_name
            else self.device
        )
        parts = [
            f"device={where}",
            f"threads={self.num_threads}",
            f"embed_batch={self.embed_batch_size}",
            f"fp16={'on' if self.embed_fp16 else 'off'}",
            f"cores={self.cpu_cores}",
            f"ram={self.ram_gb:g}GB",
        ]
        if self.overrides:
            parts.append(f"overridden={','.join(self.overrides)}")
        return "  ".join(parts)


def _torch_versions() -> tuple[str, str]:
    """Returns ``(torch version, CUDA build version)``, empty strings if absent."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return "", ""
    return str(getattr(torch, "__version__", "")), str(
        getattr(getattr(torch, "version", None), "cuda", "") or ""
    )


def profile(
    device: str | None = None,
    num_threads: int | None = None,
    embed_batch_size: int | None = None,
    embed_fp16: bool | None = None,
) -> HardwareProfile:
    """Resolves the hardware profile for this process.

    Precedence is explicit argument, then environment variable, then detection,
    so a CLI flag beats a ``.env`` and both beat the guess.

    Args:
        device: Forced device, or ``"auto"``/None to detect.
        num_threads: Forced thread count for Docling and torch.
        embed_batch_size: Forced texts per encode call.
        embed_fp16: Force half precision on or off for the embedding model.

    Returns:
        The resolved profile. Pure: nothing is applied to the process.

    """
    overrides: list[str] = []

    wanted = (device or _env("GROWNXT_DEVICE") or "auto").strip().lower()
    if wanted and wanted != "auto":
        if not wanted.startswith(_VALID_DEVICE_PREFIXES):
            logger.warning("Unknown device %r; detecting instead.", wanted)
            wanted = "auto"
        else:
            overrides.append("device")

    detected, gpu_name, vram_gb = detect_accelerator()
    if wanted == "auto":
        resolved = detected
    else:
        resolved = wanted
        if not resolved.startswith("cuda"):
            gpu_name, vram_gb = "", 0.0
        elif detected != "cuda":
            # Asked for CUDA on a machine that cannot provide it. Honour the
            # request rather than silently downgrading: torch's own error names
            # the missing piece, and a silent fall back to CPU is how a
            # fifty-hour run gets discovered at hour forty.
            logger.warning(
                "device=%s requested but torch reports no usable CUDA device.",
                resolved,
            )

    cores = physical_cores()
    ram_gb = total_ram_gb()

    threads = num_threads if num_threads and num_threads > 0 else None
    if threads is None:
        threads = _env_int("GROWNXT_NUM_THREADS")
        if threads:
            overrides.append("num_threads")
    else:
        overrides.append("num_threads")
    if not threads:
        threads = max(1, min(cores, MAX_DOCLING_THREADS))

    batch = embed_batch_size if embed_batch_size and embed_batch_size > 0 else None
    if batch is None:
        batch = _env_int("GROWNXT_EMBED_BATCH_SIZE")
        if batch:
            overrides.append("embed_batch_size")
    else:
        overrides.append("embed_batch_size")
    if not batch:
        batch = _embed_batch_size(resolved, vram_gb, ram_gb)

    # Half precision is a GPU-only win: on CPU torch emulates it and gets
    # slower, so the default is tied to the device rather than to a preference.
    fp16_default = resolved.startswith("cuda")
    if embed_fp16 is not None:
        fp16 = bool(embed_fp16)
        overrides.append("embed_fp16")
    else:
        fp16 = _env_flag("GROWNXT_EMBED_FP16", fp16_default)
        if fp16 != fp16_default:
            overrides.append("embed_fp16")
    fp16 = fp16 and resolved.startswith("cuda")

    torch_build, cuda_build = _torch_versions()
    return HardwareProfile(
        device=resolved,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        cpu_cores=cores,
        ram_gb=ram_gb,
        num_threads=int(threads),
        embed_batch_size=int(batch),
        embed_fp16=bool(fp16),
        torch_build=torch_build,
        cuda_build=cuda_build,
        gpu_present=(detected == "cuda") or nvidia_gpu_present(),
        overrides=tuple(dict.fromkeys(overrides)),
    )


def configure(
    device: str | None = None,
    num_threads: int | None = None,
    embed_batch_size: int | None = None,
    embed_fp16: bool | None = None,
    resolved: HardwareProfile | None = None,
) -> HardwareProfile:
    """Applies a profile to this process and returns it.

    Sets the environment variables the ML stack reads at import or model-build
    time, then configures torch directly for the parts that have no variable.
    Safe to call more than once and safe to call when torch is absent.

    Environment variables already set by the operator are left alone: an
    explicit ``OMP_NUM_THREADS`` in a shell or a container is a decision, not
    an accident.

    Args:
        device: See `profile`.
        num_threads: See `profile`.
        embed_batch_size: See `profile`.
        embed_fp16: See `profile`.
        resolved: A profile computed earlier, used instead of resolving again.

    Returns:
        The profile that was applied.

    """
    prof = resolved or profile(
        device=device,
        num_threads=num_threads,
        embed_batch_size=embed_batch_size,
        embed_fp16=embed_fp16,
    )

    # Docling reads both of these through pydantic-settings when it builds its
    # AcceleratorOptions; the Extractor also passes them explicitly, so this is
    # belt and braces for any Docling internal that constructs its own.
    os.environ.setdefault("DOCLING_NUM_THREADS", str(prof.num_threads))
    os.environ.setdefault("DOCLING_DEVICE", prof.device)

    # OpenMP governs the ONNX and torch CPU kernels underneath every stage.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, str(prof.num_threads))

    # HuggingFace tokenizers fork a thread pool per process and warn loudly when
    # a DataLoader forks after it; the batch never tokenises in parallel with
    # anything, so silence it rather than leave a warning per document.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch

        torch.set_num_threads(prof.num_threads)
        if prof.device.startswith("cuda"):
            # Filings are uniform in page size, so cuDNN's autotuner pays for
            # itself over hundreds of pages of identical convolution shapes.
            torch.backends.cudnn.benchmark = True
    except Exception as exc:  # noqa: BLE001 - torch is an ML extra
        logger.debug("torch thread configuration skipped: %s", exc)

    if prof.idle_gpu:
        logger.warning(
            "An NVIDIA GPU is present but torch %s has no CUDA build, so every "
            "model runs on CPU. Install a CUDA wheel (see README, 'GPU "
            "acceleration') to use the card on this machine.",
            prof.torch_build or "(not installed)",
        )
    return prof
