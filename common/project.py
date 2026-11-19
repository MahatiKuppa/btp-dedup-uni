"""Shared paths, environment handling, and small IO helpers for the pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKERS_DIR = PROJECT_ROOT / "markers"
HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"
GROUPS_CSV = PROJECT_ROOT / "data" / "lc25000_image_groups.csv"  # ground-truth group labels, all 25k images

CLASSES = ("colon_n", "colon_aca")
DEFAULT_CLASS = "colon_n"
# LC25000 class name -> HF dataset label name (organ is "colon" for both)
HF_LABELS = {"colon_n": "benign", "colon_aca": "adenocarcinomas"}


@dataclass(frozen=True)
class ClassPaths:
    """Every per-class path. colon_n keeps the original un-namespaced layout."""

    cls: str
    raw_dir: Path
    original_dir: Path  # original LC25000 file names
    embeddings_dir: Path
    results_dir: Path
    final_dir: Path
    final_manifest: Path
    group_map_csv: Path
    comparison_csv: Path
    marker_suffix: str
    file_prefix: str  # our renamed tiles: <prefix>NNNN.jpeg
    hf_label: str

    def marker(self, name: str) -> Path:
        return MARKERS_DIR / f"{name}{self.marker_suffix}.done"


def class_paths(cls: str = DEFAULT_CLASS) -> ClassPaths:
    if cls not in CLASSES:
        raise ValueError(f"unknown class {cls!r}, expected one of {CLASSES}")
    data = PROJECT_ROOT / "data"
    if cls == "colon_n":
        embeddings, results, suffix = PROJECT_ROOT / "embeddings", PROJECT_ROOT / "results", ""
    else:
        embeddings, results = PROJECT_ROOT / "embeddings" / cls, PROJECT_ROOT / "results" / cls
        suffix = f"_{cls}"
    return ClassPaths(
        cls=cls,
        raw_dir=data / "raw" / cls,
        original_dir=data / "raw_tampapath" / cls,
        embeddings_dir=embeddings,
        results_dir=results,
        final_dir=data / "final" / cls,
        final_manifest=data / "final" / f"{cls}_manifest.csv",
        group_map_csv=embeddings / "my_filename_to_group_id.csv",
        comparison_csv=results / "ground_truth_comparison.csv",
        marker_suffix=suffix,
        file_prefix=f"{cls}_",
        hf_label=HF_LABELS[cls],
    )


def add_cls_arg(parser) -> None:
    parser.add_argument("--cls", choices=CLASSES, default=DEFAULT_CLASS, help="LC25000 class, default: %(default)s")


# colon_n defaults, kept for callers that predate --cls
_DEFAULT = class_paths(DEFAULT_CLASS)
RAW_DIR = _DEFAULT.raw_dir
EMBEDDINGS_DIR = _DEFAULT.embeddings_dir
RESULTS_DIR = _DEFAULT.results_dir
ORIGINAL_DIR = _DEFAULT.original_dir
FINAL_DIR = _DEFAULT.final_dir
FINAL_MANIFEST = _DEFAULT.final_manifest
GROUP_MAP_CSV = _DEFAULT.group_map_csv
COMPARISON_CSV = _DEFAULT.comparison_csv

IMAGE_SUFFIX = ".jpeg"
N_EXPECTED = 5000

T = TypeVar("T")


def ensure_dirs(paths: ClassPaths = _DEFAULT) -> None:
    for directory in (paths.raw_dir, paths.embeddings_dir, MARKERS_DIR, HF_CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def configure_hf_env() -> None:
    """Keep the HF cache inside the project and give the hub a longer leash.

    Must be called before torch / timm / datasets are imported, since those
    read these variables at import time.
    """
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")


def die(message: str, code: int = 1) -> None:
    print(f"\nERROR: {message}\n", file=sys.stderr)
    raise SystemExit(code)


def load_hf_token() -> str:
    """Read HF_TOKEN from .env or the environment. Never hardcoded, never logged."""
    env_file = PROJECT_ROOT / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except ImportError:
        if env_file.exists() and not os.environ.get("HF_TOKEN"):
            die(f"{env_file} exists but python-dotenv is not installed: pip install python-dotenv")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        die(
            "no Hugging Face token found.\n"
            f"  1. create {PROJECT_ROOT / '.env'} containing a single line:\n"
            "       HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx\n"
            "  2. accept the model terms at https://huggingface.co/MahmoodLab/UNI\n"
            "     with the same account the token belongs to (a valid token alone\n"
            "     is not enough for a gated repo)\n"
            "  or export HF_TOKEN=... in your shell instead."
        )
    # huggingface_hub picks the token up from here; nothing is written to disk.
    os.environ["HF_TOKEN"] = token
    return token


def list_images(directory: Path = RAW_DIR) -> list[Path]:
    """Deterministic, sorted list of the raw tiles. Row order depends on this."""
    return sorted(p for p in directory.glob(f"*{IMAGE_SUFFIX}") if p.is_file())


def sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def relative_paths(paths: Iterable[Path]) -> list[str]:
    return [p.relative_to(PROJECT_ROOT).as_posix() for p in paths]


def sha256_of_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    _atomic_write(path, lambda handle: handle.write(text))


def write_json_atomic(path: Path, payload: dict) -> None:
    _atomic_write(path, lambda handle: json.dump(payload, handle, indent=2, sort_keys=True))


def _atomic_write(path: Path, writer: Callable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def with_retries(fn: Callable[[], T], *, attempts: int = 5, base_delay: float = 5.0, what: str = "operation") -> T:
    """Retry with exponential backoff. The HF hub on this box drops connections."""
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as error:  # noqa: BLE001 - surfaced after the last attempt
            last_error = error
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            print(f"  {what} failed (attempt {attempt}/{attempts}): {error}")
            print(f"  retrying in {delay:.0f}s ...")
            time.sleep(delay)
    raise RuntimeError(f"{what} failed after {attempts} attempts") from last_error
