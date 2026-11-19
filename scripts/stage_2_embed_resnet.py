#!/usr/bin/env python3
"""Stage 2 (ablation) - embed LC25000 colon-benign tiles with ResNet50 (ImageNet, 2048-d).

Run from the project root:

    python3 scripts/stage_2_embed_resnet.py

Generic-backbone counterpart to stage_2_embed.py (UNI v1), for comparing a
histopathology-specific embedding against an off-the-shelf ImageNet one on the
same images, same order, same checkpoint/resume discipline.

If data/raw/colon_n/ does not hold exactly 5000 .jpeg files, the colon/benign
subset of the HF dataset '1aurent/LC25000' is downloaded first (same logic as
stage_2_embed.py). ResNet50 weights download from PyTorch's own servers - no HF
token needed for the model itself. Embedding checkpoints every 500 images; an
interrupted run resumes where it stopped.

Outputs:
    embeddings/resnet50_full.npy    (5000, 2048) float32, L2-normalized rows
    embeddings/resnet50_paths.txt   one project-relative path per line, row-aligned
    markers/stage_2_embed_resnet.done completion marker
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.project import (  # noqa: E402
    EMBEDDINGS_DIR,
    MARKERS_DIR,
    N_EXPECTED,
    PROJECT_ROOT,
    RAW_DIR,
    configure_hf_env,
    die,
    ensure_dirs,
    list_images,
    load_hf_token,
    relative_paths,
    sha256_of_lines,
    with_retries,
    write_json_atomic,
    write_text_atomic,
)

MODEL_NAME = "torchvision/resnet50-imagenet1k-v2"
EMBED_DIM = 2048

DATASET_ID = "1aurent/LC25000"
ORGAN = "colon"
LABEL = "benign"

CHECKPOINT_EVERY = 500
DEFAULT_BATCH_SIZE = 8
DEFAULT_WORKERS = 4

FINAL_PATH = EMBEDDINGS_DIR / "resnet50_full.npy"
PATHS_PATH = EMBEDDINGS_DIR / "resnet50_paths.txt"
PARTIAL_PATH = EMBEDDINGS_DIR / "_partial_resnet50.npy"
META_PATH = EMBEDDINGS_DIR / "_partial_resnet50.meta.json"
MARKER_PATH = MARKERS_DIR / "stage_2_embed_resnet.done"


# --------------------------------------------------------------------------- #
# idempotency
# --------------------------------------------------------------------------- #
def already_done() -> bool:
    if not (MARKER_PATH.exists() and FINAL_PATH.exists()):
        return False
    try:
        shape = np.load(FINAL_PATH, mmap_mode="r").shape
    except Exception:  # noqa: BLE001 - a corrupt output just means "not done"
        return False
    return shape == (N_EXPECTED, EMBED_DIM)


# --------------------------------------------------------------------------- #
# data (identical logic to stage_2_embed.py - same images, same directory)
# --------------------------------------------------------------------------- #
def ensure_images(token: str) -> list[Path]:
    images = list_images()
    if len(images) == N_EXPECTED:
        print(f"[data] found {N_EXPECTED} images in {RAW_DIR.relative_to(PROJECT_ROOT)}/")
        return images

    if len(images) > N_EXPECTED:
        die(
            f"{RAW_DIR} holds {len(images)} .jpeg files, expected exactly {N_EXPECTED}. "
            "Something else wrote into that directory; clean it and re-run."
        )
    if images:
        print(f"[data] {len(images)}/{N_EXPECTED} images present - resuming download")
    else:
        print(f"[data] {RAW_DIR.relative_to(PROJECT_ROOT)}/ is empty - downloading {DATASET_ID}")

    download_colon_benign(token)

    images = list_images()
    if len(images) != N_EXPECTED:
        die(
            f"after download, {RAW_DIR} holds {len(images)} .jpeg files, expected {N_EXPECTED}. "
            "Delete that directory and re-run to download from scratch."
        )
    return images


def download_colon_benign(token: str) -> None:
    from datasets import Image as HFImage
    from datasets import load_dataset
    from tqdm import tqdm

    print(f"[data] fetching full {DATASET_ID} (~1.9 GB; parquet shards mix organs, so no partial fetch)")
    dataset = with_retries(
        lambda: load_dataset(DATASET_ID, split="train", token=token),
        what=f"load_dataset({DATASET_ID})",
    )

    # Resolve class ids by name; the integer encoding is not guaranteed stable.
    organ_id = dataset.features["organ"].str2int(ORGAN)
    label_id = dataset.features["label"].str2int(LABEL)

    # Raw bytes only: writing the original JPEG avoids re-encode artifacts.
    dataset = dataset.cast_column("image", HFImage(decode=False))
    subset = dataset.filter(
        lambda organs, labels: [o == organ_id and l == label_id for o, l in zip(organs, labels)],
        input_columns=["organ", "label"],
        batched=True,
        batch_size=2000,
        desc=f"filter {ORGAN}/{LABEL}",
    )
    if len(subset) != N_EXPECTED:
        die(f"{ORGAN}/{LABEL} subset of {DATASET_ID} has {len(subset)} rows, expected {N_EXPECTED}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for index, record in enumerate(tqdm(subset, desc="writing jpegs", unit="img")):
        destination = RAW_DIR / f"colon_n_{index:04d}.jpeg"
        if destination.exists() and destination.stat().st_size > 0:
            skipped += 1
            continue
        payload = record["image"]["bytes"]
        if payload is None:
            payload = Path(record["image"]["path"]).read_bytes()
        partial = destination.with_name(destination.name + ".part")
        partial.write_bytes(payload)
        partial.replace(destination)
        written += 1
    print(f"[data] wrote {written} jpegs, {skipped} already present")


class TileDataset:
    """Map-style dataset that yields (tensor, absolute row index)."""

    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as handle:
            return self.transform(handle.convert("RGB")), index


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_model():
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    print(f"[model] loading {MODEL_NAME} (torchvision downloads its own weights, no HF token needed)")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = with_retries(lambda: resnet50(weights=weights), what=f"resnet50(weights={weights})")
    model.fc = torch.nn.Identity()  # avgpool output (2048-d), not the 1000-class logits
    model.eval()

    transform = weights.transforms()
    meta = transform.__dict__ if hasattr(transform, "__dict__") else {}
    print(
        f"[model] {EMBED_DIM}-d | crop_size={getattr(transform, 'crop_size', '?')} "
        f"resize_size={getattr(transform, 'resize_size', '?')} "
        f"mean={getattr(transform, 'mean', '?')} std={getattr(transform, 'std', '?')}"
    )
    return model, transform


def make_loader(images: list[Path], transform, indices: range, batch_size: int, workers: int):
    import torch

    dataset = torch.utils.data.Subset(TileDataset(images, transform), indices)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=False,
    )


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #
def resume_position(paths_digest: str, total: int) -> int:
    import json

    have_partial, have_meta = PARTIAL_PATH.exists(), META_PATH.exists()
    if not (have_partial and have_meta):
        if have_partial or have_meta:
            print("[ckpt] found only half of a checkpoint pair - discarding it and starting over")
            PARTIAL_PATH.unlink(missing_ok=True)
            META_PATH.unlink(missing_ok=True)
        return 0

    try:
        meta = json.loads(META_PATH.read_text())
    except (OSError, ValueError) as error:
        die(f"cannot read {META_PATH}: {error}. Delete {PARTIAL_PATH.name} and {META_PATH.name} to start over.")

    mismatches = [
        name
        for name, ok in (
            ("image list", meta.get("paths_sha256") == paths_digest),
            ("image count", meta.get("n_total") == total),
            ("model", meta.get("model") == MODEL_NAME),
            ("embedding dim", meta.get("dim") == EMBED_DIM),
        )
        if not ok
    ]
    if mismatches:
        die(
            f"the checkpoint in {EMBEDDINGS_DIR.relative_to(PROJECT_ROOT)}/ was made with a different "
            f"{', '.join(mismatches)}. Refusing to mix embeddings.\n"
            f"  Delete {PARTIAL_PATH.name} and {META_PATH.name} to start over."
        )

    n_done = int(meta.get("n_done", 0))
    if not 0 <= n_done <= total:
        die(f"checkpoint claims n_done={n_done} of {total}. Delete {PARTIAL_PATH.name} and {META_PATH.name}.")
    return n_done


def open_partial(total: int, resuming: bool):
    if not resuming:
        return np.lib.format.open_memmap(PARTIAL_PATH, mode="w+", dtype=np.float32, shape=(total, EMBED_DIM))
    array = np.lib.format.open_memmap(PARTIAL_PATH, mode="r+")
    if array.shape != (total, EMBED_DIM) or array.dtype != np.float32:
        die(f"{PARTIAL_PATH.name} has shape {array.shape} {array.dtype}, expected ({total}, {EMBED_DIM}) float32")
    return array


def save_checkpoint(array, n_done: int, total: int, paths_digest: str) -> None:
    array.flush()
    # Meta is written only after the rows it vouches for are on disk.
    write_json_atomic(
        META_PATH,
        {
            "n_done": int(n_done),
            "n_total": int(total),
            "dim": EMBED_DIM,
            "model": MODEL_NAME,
            "paths_sha256": paths_digest,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #
def run_embedding(images: list[Path], model, transform, batch_size: int, workers: int) -> None:
    import torch
    from tqdm import tqdm

    total = len(images)
    rel_paths = relative_paths(images)
    paths_digest = sha256_of_lines(rel_paths)
    write_text_atomic(PATHS_PATH, "\n".join(rel_paths) + "\n")
    print(f"[out] wrote {PATHS_PATH.relative_to(PROJECT_ROOT)} ({total} lines, project-relative)")

    start = resume_position(paths_digest, total)
    array = open_partial(total, resuming=start > 0)
    if start:
        print(f"[ckpt] resuming at image {start}/{total}")
    if start == 0:
        save_checkpoint(array, 0, total, paths_digest)

    position = start
    last_saved = start
    began = time.time()

    if start < total:
        loader = make_loader(images, transform, range(start, total), batch_size, workers)
        try:
            with torch.inference_mode(), tqdm(
                total=total, initial=start, unit="img", desc="embedding", dynamic_ncols=True, smoothing=0.05
            ) as bar:
                for tensors, indices in loader:
                    features = model(tensors).to(torch.float32).numpy()
                    rows = indices.numpy()
                    if rows[0] != position:
                        die(f"loader out of order: expected row {position}, got {rows[0]}")
                    array[rows] = features
                    position = int(rows[-1]) + 1
                    bar.update(len(rows))

                    if position - last_saved >= CHECKPOINT_EVERY:
                        save_checkpoint(array, position, total, paths_digest)
                        last_saved = position
                        bar.set_postfix_str(f"ckpt@{position}")
        except KeyboardInterrupt:
            save_checkpoint(array, position, total, paths_digest)
            print(f"\n[ckpt] interrupted - progress saved at {position}/{total}. Re-run the same command to resume.")
            raise SystemExit(130)

        save_checkpoint(array, total, total, paths_digest)
        processed = total - start
        elapsed = time.time() - began
        print(f"[time] {processed} images in {elapsed / 60:.1f} min ({elapsed / processed:.2f} s/img)")

    finalize(array, total)


def finalize(array, total: int) -> None:
    matrix = np.array(array, dtype=np.float32)
    del array
    gc.collect()

    if not np.all(np.isfinite(matrix)):
        die("non-finite values in raw embeddings. Delete the checkpoint files and re-run.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_rows = np.flatnonzero(norms[:, 0] == 0)
    if zero_rows.size:
        die(f"{zero_rows.size} all-zero embedding rows (first: {zero_rows[:5].tolist()}); checkpoint is inconsistent.")
    matrix /= norms

    assert matrix.shape == (N_EXPECTED, EMBED_DIM), f"shape {matrix.shape} != ({N_EXPECTED}, {EMBED_DIM})"
    row_norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(row_norms, 1.0, atol=1e-5), f"row norms deviate from 1 by up to {np.abs(row_norms - 1).max():.3e}"

    np.save(FINAL_PATH, matrix)
    reloaded = np.load(FINAL_PATH, mmap_mode="r")
    assert reloaded.shape == (N_EXPECTED, EMBED_DIM), f"saved shape {reloaded.shape}"
    print(f"[out] wrote {FINAL_PATH.relative_to(PROJECT_ROOT)} {matrix.shape} {matrix.dtype}")
    print(f"[check] shape OK | row norms in [{row_norms.min():.6f}, {row_norms.max():.6f}]")

    write_marker(total)
    PARTIAL_PATH.unlink(missing_ok=True)
    META_PATH.unlink(missing_ok=True)


def write_marker(total: int) -> None:
    import torch
    import torchvision

    write_json_atomic(
        MARKER_PATH,
        {
            "stage": "stage_2_embed_resnet",
            "completed": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": MODEL_NAME,
            "dim": EMBED_DIM,
            "n_images": total,
            "embeddings": FINAL_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "paths": PATHS_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "l2_normalized": True,
            "torchvision_version": torchvision.__version__,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
    )
    print(f"[out] wrote {MARKER_PATH.relative_to(PROJECT_ROOT)}")


def benchmark(images: list[Path], model, transform, batch_size: int, workers: int) -> None:
    """Time a few images and project the full run. Writes nothing."""
    import torch

    loader = make_loader(images, transform, range(len(images)), batch_size, workers)
    timed = 0
    elapsed = 0.0
    with torch.inference_mode():
        for batch_number, (tensors, _) in enumerate(loader):
            began = time.time()
            model(tensors)
            if batch_number == 0:
                continue  # warm-up batch excluded from timing
            elapsed += time.time() - began
            timed += len(tensors)
    if not timed:
        die(f"--limit must exceed one batch ({batch_size}) to get a timing")
    per_image = elapsed / timed
    print(f"[bench] model forward: {per_image:.3f} s/img over {timed} images (warm-up excluded, decode not included)")
    print(f"[bench] projected full run: ~{per_image * N_EXPECTED / 60:.0f} min + JPEG decode overhead")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="default: %(default)s")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="DataLoader workers, default: %(default)s")
    parser.add_argument("--threads", type=int, default=None, help="torch intra-op threads, default: cores - workers")
    parser.add_argument("--limit", type=int, default=None, help="benchmark only: time N images, write nothing")
    args = parser.parse_args()

    ensure_dirs()
    configure_hf_env()

    if args.limit is None and already_done():
        print("already done, skipping")
        return 0

    token = load_hf_token()
    images = ensure_images(token)

    import torch

    threads = args.threads or max(1, (os.cpu_count() or 4) - args.workers)
    torch.set_num_threads(threads)
    print(f"[cpu] torch {torch.__version__} | threads={threads} | workers={args.workers} | batch={args.batch_size}")

    model, transform = build_model()

    if args.limit is not None:
        benchmark(images[: args.limit], model, transform, args.batch_size, args.workers)
        return 0

    run_embedding(images, model, transform, args.batch_size, args.workers)
    print("[done] stage 2 (resnet) complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
