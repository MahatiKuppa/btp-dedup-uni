"""Shared paths, environment handling, and small IO helpers for the pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "colon_n"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
MARKERS_DIR = PROJECT_ROOT / "markers"
RESULTS_DIR = PROJECT_ROOT / "results"
HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"
ORIGINAL_DIR = PROJECT_ROOT / "data" / "raw_tampapath" / "colon_n"  # original LC25000 file names
GROUPS_CSV = PROJECT_ROOT / "data" / "lc25000_image_groups.csv"  # ground-truth group labels, all 25k images
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "colon_n"
FINAL_MANIFEST = PROJECT_ROOT / "data" / "final" / "colon_n_manifest.csv"
GROUP_MAP_CSV = EMBEDDINGS_DIR / "my_filename_to_group_id.csv"
COMPARISON_CSV = RESULTS_DIR / "ground_truth_comparison.csv"

IMAGE_SUFFIX = ".jpeg"
N_EXPECTED = 5000

T = TypeVar("T")


def ensure_dirs() -> None:
    for directory in (RAW_DIR, EMBEDDINGS_DIR, MARKERS_DIR, HF_CACHE_DIR):
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
