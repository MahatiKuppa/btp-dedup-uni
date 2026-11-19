#!/usr/bin/env python3
"""Stage 3 - KMeans (k=250) on the UNI/ResNet50 embeddings, with a UMAP view and PCA ablations.

DEPENDENCIES - install these before running:

    pip3 install --break-system-packages scikit-learn matplotlib umap-learn

(scipy, numba and llvmlite arrive as dependencies of those three.)

Run from the project root (add --cls colon_aca for the adenocarcinoma class; default colon_n):

    python3 scripts/stage_3_kmeans.py                            # UNI, 1024-d
    python3 scripts/stage_3_kmeans.py --embedding uni_pca128      # UNI, 128-d PCA
    python3 scripts/stage_3_kmeans.py --embedding resnet          # ResNet50, 2048-d
    python3 scripts/stage_3_kmeans.py --embedding resnet_pca128   # ResNet50, 128-d PCA
    python3 scripts/stage_3_kmeans.py --force                     # rerun despite the marker

`uni`/`uni_pca128` need embeddings/uni1_full.npy (scripts/stage_2_embed.py).
`resnet`/`resnet_pca128` need embeddings/resnet50_full.npy (scripts/stage_2_embed_resnet.py).

Diagnostic only: the 5000 images are augmentations of 250 originals, so k=250 is the
"one cluster per original" hypothesis. The outputs are there to show whether those
clusters are genuinely separated or just cells carved out of one continuous blob.

Outputs (per arm, in results/kmeans_{embedding}/):
    labels.npy          (5000,)   int32 cluster assignment per image
    centers.npy         (250, d)  float32 centroids
    cluster_sizes.csv   cluster_id, n_members
    metrics.csv         one row: intra/inter variance, separation ratio, size stats
    umap_coords.npy     (5000, 2) float32, cached so re-plotting skips UMAP
    umap_clusters.png   scatter of all 5000 points, colored by cluster
    spot_check.png      6 member images each from a small, average and large cluster
    markers/stage_3_kmeans_{embedding}.done
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.project import (  # noqa: E402
    DEFAULT_CLASS,
    N_EXPECTED,
    PROJECT_ROOT,
    add_cls_arg,
    class_paths,
    die,
    ensure_dirs,
    write_json_atomic,
)

PIP_HINT = "pip3 install --break-system-packages scikit-learn matplotlib umap-learn"

K = 250
RANDOM_STATE = 42
N_INIT = 10
PCA_COMPONENTS = 128
SMALL_CLUSTER = 5
LARGE_CLUSTER = 40
SPOT_CHECK_IMAGES = 6
THUMBNAIL_PX = 256


def bind(cls: str) -> None:
    """Point the per-class module constants at `cls` (colon_n keeps the original layout)."""
    global P, EMBEDDINGS_DIR, RESULTS_DIR, PATHS_PATH, SOURCES, PCA_ARMS
    P = class_paths(cls)
    EMBEDDINGS_DIR, RESULTS_DIR = P.embeddings_dir, P.results_dir
    PATHS_PATH = EMBEDDINGS_DIR / "paths.txt"  # shared: identical across embedding sources

    # Base (non-PCA) arms: where the raw embedding matrix lives.
    SOURCES = {
        "uni": {
            "path": EMBEDDINGS_DIR / "uni1_full.npy",
            "dim": 1024,
            "label": "UNI",
            "hint": f"python3 scripts/stage_2_embed.py --cls {cls}",
        },
        "resnet": {
            "path": EMBEDDINGS_DIR / "resnet50_full.npy",
            "dim": 2048,
            "label": "ResNet50",
            "hint": f"python3 scripts/stage_2_embed_resnet.py --cls {cls}",
        },
    }

    # PCA arms: which base arm they reduce, and where the cached reduction lives.
    PCA_ARMS = {
        "uni_pca128": {
            "base": "uni",
            "path": EMBEDDINGS_DIR / "uni1_pca128.npy",
            "meta": EMBEDDINGS_DIR / "uni1_pca128.meta.json",
        },
        "resnet_pca128": {
            "base": "resnet",
            "path": EMBEDDINGS_DIR / "resnet50_pca128.npy",
            "meta": EMBEDDINGS_DIR / "resnet50_pca128.meta.json",
        },
    }


bind(DEFAULT_CLASS)

# Neutral ink: text never wears a series color.
INK = "#1f2328"
INK_MUTED = "#57606a"


def require(module_name: str):
    """Import a heavy optional dependency, or exit naming the install command."""
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        die(f"missing dependency '{module_name}' ({error}).\n  install with: {PIP_HINT}")


# --------------------------------------------------------------------------- #
# paths per arm
# --------------------------------------------------------------------------- #
def output_dir(arm: str) -> Path:
    return RESULTS_DIR / f"kmeans_{arm}"


def marker_path(arm: str) -> Path:
    return P.marker(f"stage_3_kmeans_{arm}")


# --------------------------------------------------------------------------- #
# idempotency
# --------------------------------------------------------------------------- #
def already_done(arm: str) -> bool:
    labels_path = output_dir(arm) / "labels.npy"
    if not (marker_path(arm).exists() and labels_path.exists()):
        return False
    try:
        shape = np.load(labels_path, mmap_mode="r").shape
    except Exception:  # noqa: BLE001 - a corrupt output just means "not done"
        return False
    return shape == (N_EXPECTED,)


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def load_source(base_arm: str) -> np.ndarray:
    source = SOURCES[base_arm]
    path, dim = source["path"], source["dim"]
    if not path.exists():
        die(f"{path} not found. Run stage 2 first: {source['hint']}")
    features = np.load(path)
    if features.shape != (N_EXPECTED, dim):
        die(f"{path.name} has shape {features.shape}, expected ({N_EXPECTED}, {dim})")
    return np.ascontiguousarray(features, dtype=np.float32)


def load_pca(arm: str) -> np.ndarray:
    spec = PCA_ARMS[arm]
    pca_path, meta_path, base_arm = spec["path"], spec["meta"], spec["base"]

    if pca_path.exists():
        features = np.load(pca_path)
        if features.shape != (N_EXPECTED, PCA_COMPONENTS):
            die(
                f"{pca_path.name} has shape {features.shape}, expected ({N_EXPECTED}, {PCA_COMPONENTS}). "
                f"Delete it to refit PCA."
            )
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            cumulative = meta.get("explained_variance_cumulative")
            print(f"[pca] loaded cached {pca_path.name} | cumulative explained variance {cumulative:.4f}")
        else:
            print(f"[pca] loaded cached {pca_path.name} | explained variance unknown ({meta_path.name} missing)")
        return np.ascontiguousarray(features, dtype=np.float32)

    decomposition = require("sklearn.decomposition")
    source_path, source_dim = SOURCES[base_arm]["path"], SOURCES[base_arm]["dim"]
    full = load_source(base_arm)
    print(f"[pca] fitting PCA({PCA_COMPONENTS}) on {full.shape} ...")
    began = time.time()
    pca = decomposition.PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    # Raw projection, deliberately not re-normalized: PCA preserves the Euclidean
    # geometry of the centered originals, keeping this a clean dimensionality ablation.
    features = np.ascontiguousarray(pca.fit_transform(full), dtype=np.float32)
    ratios = pca.explained_variance_ratio_
    cumulative = float(ratios.sum())

    np.save(pca_path, features)
    write_json_atomic(
        meta_path,
        {
            "source": source_path.relative_to(PROJECT_ROOT).as_posix(),
            "source_shape": [int(x) for x in full.shape],
            "n_components": PCA_COMPONENTS,
            "random_state": RANDOM_STATE,
            "l2_renormalized": False,
            "explained_variance_cumulative": cumulative,
            "explained_variance_ratio": [float(x) for x in ratios],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    print(f"[pca] fitted in {time.time() - began:.1f}s | wrote {pca_path.relative_to(PROJECT_ROOT)} {features.shape}")
    print(
        f"[pca] cumulative explained variance: {cumulative:.4f} "
        f"({cumulative * 100:.2f}% of the original {source_dim}-d variance in {PCA_COMPONENTS} components)"
    )
    return features


def arm_label(arm: str) -> str:
    base = arm if arm in SOURCES else PCA_ARMS[arm]["base"]
    return SOURCES[base]["label"]


def load_features(arm: str) -> np.ndarray:
    if arm in SOURCES:
        features = load_source(arm)
        print(f"[data] loaded {SOURCES[arm]['path'].relative_to(PROJECT_ROOT)} {features.shape} (L2-normalized rows)")
        return features
    return load_pca(arm)


def load_paths() -> list[Path]:
    if not PATHS_PATH.exists():
        die(f"{PATHS_PATH} not found. Run stage 2 first.")
    lines = [line.strip() for line in PATHS_PATH.read_text().splitlines() if line.strip()]
    if len(lines) != N_EXPECTED:
        die(f"{PATHS_PATH.name} has {len(lines)} lines, expected {N_EXPECTED}")
    return [PROJECT_ROOT / line for line in lines]


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
def run_kmeans(features: np.ndarray):
    cluster = require("sklearn.cluster")
    print(f"[kmeans] k={K} random_state={RANDOM_STATE} n_init={N_INIT} on {features.shape} ...")
    began = time.time()
    model = cluster.KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = model.fit_predict(features).astype(np.int32)
    centers = np.ascontiguousarray(model.cluster_centers_, dtype=np.float32)
    print(f"[kmeans] converged in {time.time() - began:.1f}s | inertia {model.inertia_:.4f}")

    sizes = np.bincount(labels, minlength=K)
    non_empty = int((sizes > 0).sum())
    if non_empty != K:
        print(f"[warn] {non_empty} non-empty clusters, expected exactly {K} ({K - non_empty} collapsed to empty)")
    assert int(sizes.sum()) == N_EXPECTED, f"{int(sizes.sum())} images across clusters, expected {N_EXPECTED}"
    assert labels.shape == (N_EXPECTED,), f"labels shape {labels.shape}"
    assert centers.shape == (K, features.shape[1]), f"centers shape {centers.shape}"

    return labels, centers, sizes, float(model.inertia_)


def compute_metrics(centers: np.ndarray, inertia: float) -> tuple[float, float, float]:
    distance = require("scipy.spatial.distance")
    # KMeans here uses Euclidean distance, UMAP uses cosine - equivalent up to a monotonic
    # transform on L2-normalized vectors (the "uni" arm), but not on the raw uni_pca128 projection.
    # Mean squared distance from each image to its own center.
    intra = inertia / N_EXPECTED
    # Mean squared distance over the K*(K-1)/2 distinct center pairs.
    inter = float(distance.pdist(centers.astype(np.float64), metric="sqeuclidean").mean())
    ratio = inter / intra if intra > 0 else float("inf")
    return intra, inter, ratio


def report(arm: str, features: np.ndarray, sizes: np.ndarray, intra: float, inter: float, ratio: float) -> dict:
    n_small = int((sizes < SMALL_CLUSTER).sum())
    n_large = int((sizes > LARGE_CLUSTER).sum())

    print(f"\n[sizes] min={int(sizes.min())} max={int(sizes.max())} "
          f"mean={sizes.mean():.2f} std={sizes.std():.2f}")
    print(f"[sizes] {n_small} clusters with <{SMALL_CLUSTER} members | "
          f"{n_large} clusters with >{LARGE_CLUSTER} members")
    print(f"[metrics] intra-cluster variance : {intra:.6f}  (mean squared distance to own center)")
    print(f"[metrics] inter-cluster variance : {inter:.6f}  (mean squared distance between centers)")
    print(f"[metrics] separation ratio       : {ratio:.4f}  (inter/intra, higher is better)\n")

    return {
        "embedding": arm,
        "dim": int(features.shape[1]),
        "k": K,
        "n_images": N_EXPECTED,
        "intra_variance": intra,
        "inter_variance": inter,
        "separation_ratio": ratio,
        "size_min": int(sizes.min()),
        "size_max": int(sizes.max()),
        "size_mean": float(sizes.mean()),
        "size_std": float(sizes.std()),
        f"n_small_lt{SMALL_CLUSTER}": n_small,
        f"n_large_gt{LARGE_CLUSTER}": n_large,
    }


def write_tables(out_dir: Path, sizes: np.ndarray, metrics: dict) -> None:
    pandas = require("pandas")
    sizes_csv = out_dir / "cluster_sizes.csv"
    metrics_csv = out_dir / "metrics.csv"
    pandas.DataFrame({"cluster_id": np.arange(K, dtype=np.int32), "n_members": sizes.astype(np.int32)}).to_csv(
        sizes_csv, index=False
    )
    pandas.DataFrame([metrics]).to_csv(metrics_csv, index=False)
    print(f"[out] wrote {sizes_csv.relative_to(PROJECT_ROOT)} and {metrics_csv.relative_to(PROJECT_ROOT)}")


# --------------------------------------------------------------------------- #
# UMAP
# --------------------------------------------------------------------------- #
def umap_coords(features: np.ndarray, coords_path: Path) -> np.ndarray:
    if coords_path.exists():
        coords = np.load(coords_path)
        if coords.shape == (N_EXPECTED, 2):
            print(f"[umap] loaded cached {coords_path.relative_to(PROJECT_ROOT)}")
            return coords
        print(f"[umap] cached coords have shape {coords.shape}, expected ({N_EXPECTED}, 2) - recomputing")

    umap_module = require("umap")
    print("[umap] n_neighbors=15 min_dist=0.1 metric=cosine random_state=42 "
          "(first call pays a numba JIT compile; random_state forces single-threaded layout)")
    began = time.time()
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    coords = np.ascontiguousarray(reducer.fit_transform(features), dtype=np.float32)
    np.save(coords_path, coords)
    print(f"[umap] done in {time.time() - began:.1f}s | wrote {coords_path.relative_to(PROJECT_ROOT)} {coords.shape}")
    return coords


def cluster_colors(pyplot) -> np.ndarray:
    """One color per cluster id.

    250 clusters far exceed any qualitative palette, so the 60 distinct hues of
    tab20/tab20b/tab20c are assigned under a fixed permutation: neighbouring ids
    get different colors, but color does not identify a cluster. The plot is read
    for blob structure, which is why there is no legend.
    """
    base = np.concatenate([np.asarray(pyplot.get_cmap(name).colors) for name in ("tab20", "tab20b", "tab20c")])
    slots = np.random.default_rng(0).permutation(K) % len(base)
    return base[slots]


def plot_umap(arm: str, coords: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    matplotlib = require("matplotlib")
    matplotlib.use("Agg")
    pyplot = require("matplotlib.pyplot")

    colors = cluster_colors(pyplot)
    figure, axes = pyplot.subplots(figsize=(12, 9), dpi=150)
    figure.patch.set_facecolor("white")
    axes.set_facecolor("white")

    axes.scatter(coords[:, 0], coords[:, 1], c=colors[labels], s=4, alpha=0.75, linewidths=0)

    # Recessive frame: the marks carry the information, not the chrome.
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)

    figure.suptitle(
        f"UMAP of the {arm} {arm_label(arm)} embeddings, colored by KMeans cluster (k={K})",
        fontsize=15,
        color=INK,
        x=0.5,
        y=0.96,
    )
    axes.set_title(
        f"{N_EXPECTED} images  ·  colors repeat across the {K} clusters and do not identify them  ·  "
        "n_neighbors=15, min_dist=0.1, cosine",
        fontsize=10,
        color=INK_MUTED,
        pad=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(out_path, dpi=150, facecolor="white")
    pyplot.close(figure)
    print(f"[out] wrote {out_path.relative_to(PROJECT_ROOT)} (1800x1350)")


# --------------------------------------------------------------------------- #
# spot check
# --------------------------------------------------------------------------- #
def pick_spot_clusters(sizes: np.ndarray) -> list[tuple[int, str]]:
    """Smallest, closest-to-average and largest cluster, as distinct ids."""
    by_size = np.argsort(sizes, kind="stable")
    smallest = int(by_size[0])
    largest = int(by_size[-1])
    target = N_EXPECTED / K
    by_distance = np.argsort(np.abs(sizes - target), kind="stable")
    average = next((int(c) for c in by_distance if c not in (smallest, largest)), int(by_distance[0]))
    return [(smallest, "smallest"), (average, f"~average, target {target:.0f}"), (largest, "largest")]


def plot_spot_check(arm: str, labels: np.ndarray, sizes: np.ndarray, paths: list[Path], out_path: Path) -> None:
    matplotlib = require("matplotlib")
    matplotlib.use("Agg")
    pyplot = require("matplotlib.pyplot")
    image_module = require("PIL.Image")

    picks = pick_spot_clusters(sizes)
    print("[spot] clusters: " + ", ".join(f"{cid} (n={int(sizes[cid])}, {tag})" for cid, tag in picks))

    figure = pyplot.figure(figsize=(12, 7), dpi=150)
    figure.patch.set_facecolor("white")
    rows = figure.subfigures(len(picks), 1, hspace=0.08)

    for row, (cluster_id, tag) in zip(np.atleast_1d(rows), picks):
        members = np.flatnonzero(labels == cluster_id)[:SPOT_CHECK_IMAGES]
        row.suptitle(
            f"cluster {cluster_id}  —  n={int(sizes[cluster_id])}  ({tag})",
            fontsize=11,
            color=INK,
            x=0.02,
            ha="left",
        )
        axes = np.atleast_1d(row.subplots(1, SPOT_CHECK_IMAGES))
        for column, axis in enumerate(axes):
            axis.set_xticks([])
            axis.set_yticks([])
            if column >= len(members):
                axis.axis("off")
                continue
            index = int(members[column])
            with image_module.open(paths[index]) as handle:
                handle.draft("RGB", (THUMBNAIL_PX, THUMBNAIL_PX))  # cheap JPEG-native downscale
                thumbnail = handle.convert("RGB")
                thumbnail.thumbnail((THUMBNAIL_PX, THUMBNAIL_PX))
                axis.imshow(thumbnail)
            axis.set_xlabel(paths[index].stem.replace(P.file_prefix, "#"), fontsize=7, color=INK_MUTED)
            for spine in axis.spines.values():
                spine.set_color("#d8dee4")

    figure.suptitle(
        f"Cluster spot check — {arm} embeddings, up to {SPOT_CHECK_IMAGES} members per cluster",
        fontsize=14,
        color=INK,
        y=0.99,
    )
    figure.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    pyplot.close(figure)
    print(f"[out] wrote {out_path.relative_to(PROJECT_ROOT)}")


# --------------------------------------------------------------------------- #
def write_marker(arm: str, metrics: dict, out_dir: Path) -> None:
    versions = {}
    for label, module_name in (("sklearn", "sklearn"), ("umap", "umap"), ("matplotlib", "matplotlib"),
                               ("pandas", "pandas"), ("scipy", "scipy")):
        try:
            versions[f"{label}_version"] = getattr(importlib.import_module(module_name), "__version__", "?")
        except ImportError:
            versions[f"{label}_version"] = "not installed"

    payload = {
        "stage": "stage_3_kmeans",
        "class": P.cls,
        "embedding": arm,
        "completed": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "k": K,
        "random_state": RANDOM_STATE,
        "n_init": N_INIT,
        "results_dir": out_dir.relative_to(PROJECT_ROOT).as_posix(),
        "numpy_version": np.__version__,
        **{key: metrics[key] for key in ("dim", "intra_variance", "inter_variance", "separation_ratio")},
        **versions,
    }
    path = marker_path(arm)
    write_json_atomic(path, payload)
    print(f"[out] wrote {path.relative_to(PROJECT_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--embedding",
        choices=("uni", "uni_pca128", "resnet", "resnet_pca128"),
        default="uni",
        help="default: %(default)s",
    )
    add_cls_arg(parser)
    parser.add_argument("--force", action="store_true", help="rerun even if the completion marker exists")
    args = parser.parse_args()
    arm = args.embedding
    bind(args.cls)

    ensure_dirs(P)
    out_dir = output_dir(arm)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.force and already_done(arm):
        print(f"already done ({arm}), skipping")
        return 0

    features = load_features(arm)
    labels, centers, sizes, inertia = run_kmeans(features)

    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "centers.npy", centers)
    print(f"[out] wrote {(out_dir / 'labels.npy').relative_to(PROJECT_ROOT)} {labels.shape} "
          f"and {(out_dir / 'centers.npy').relative_to(PROJECT_ROOT)} {centers.shape}")

    intra, inter, ratio = compute_metrics(centers, inertia)
    metrics = report(arm, features, sizes, intra, inter, ratio)
    write_tables(out_dir, sizes, metrics)

    coords = umap_coords(features, out_dir / "umap_coords.npy")
    plot_umap(arm, coords, labels, out_dir / "umap_clusters.png")
    plot_spot_check(arm, labels, sizes, load_paths(), out_dir / "spot_check.png")

    write_marker(arm, metrics, out_dir)
    print(f"[done] stage 3 ({arm}) complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
