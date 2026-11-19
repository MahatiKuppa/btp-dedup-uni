#!/usr/bin/env python3
"""Stage 5 - compare the KMeans arms against the ground-truth group ids.

Run from the project root (needs stage 3 results for all four arms and stage 4):

    python3 scripts/stage_5_compare_ground_truth.py
    python3 scripts/stage_5_compare_ground_truth.py --force     # rerun despite the marker

For each arm in results/kmeans_{uni,uni_pca128,resnet,resnet_pca128}/labels.npy computes
adjusted Rand index and Fowlkes-Mallows score against group_id. Read-only on all inputs.

Outputs:
    results/ground_truth_comparison.csv   one row per arm
    markers/stage_5_compare_ground_truth.done
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.project import (  # noqa: E402
    COMPARISON_CSV,
    EMBEDDINGS_DIR,
    GROUP_MAP_CSV,
    MARKERS_DIR,
    N_EXPECTED,
    PROJECT_ROOT,
    RESULTS_DIR,
    die,
    ensure_dirs,
    write_json_atomic,
)

ARMS = ("kmeans_uni", "kmeans_uni_pca128", "kmeans_resnet", "kmeans_resnet_pca128")
PATHS_PATH = EMBEDDINGS_DIR / "paths.txt"
MARKER = MARKERS_DIR / "stage_5_compare_ground_truth.done"


def load_group_ids() -> np.ndarray:
    """group_id per row, verified to follow embeddings/paths.txt row order."""
    if not GROUP_MAP_CSV.exists():
        die(f"missing {GROUP_MAP_CSV.relative_to(PROJECT_ROOT)}: run scripts/stage_4_match_groups.py")
    mapping = pd.read_csv(GROUP_MAP_CSV)
    paths = [Path(line.strip()).name for line in PATHS_PATH.read_text().splitlines() if line.strip()]
    if len(mapping) != N_EXPECTED or len(paths) != N_EXPECTED:
        die(f"expected {N_EXPECTED} rows, mapping={len(mapping)} paths.txt={len(paths)}")
    if mapping["my_filename"].tolist() != paths:
        die("my_filename order in the mapping does not match embeddings/paths.txt")
    if mapping["group_id"].isna().any():
        die("mapping has missing group_id")
    print(f"[data] mapping aligned to embeddings/paths.txt ({len(mapping)} rows, "
          f"{mapping['group_id'].nunique()} distinct group_id)")
    return mapping["group_id"].to_numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rerun even if the completion marker exists")
    args = parser.parse_args()

    ensure_dirs()
    if not args.force and MARKER.exists() and COMPARISON_CSV.exists():
        print("already done (stage 5), skipping")
        return 0

    truth = load_group_ids()
    rows = []
    for arm in ARMS:
        path = RESULTS_DIR / arm / "labels.npy"
        if not path.exists():
            die(f"missing {path.relative_to(PROJECT_ROOT)}: run stage 3 for this arm")
        labels = np.load(path)
        if labels.shape != (N_EXPECTED,):
            die(f"{path.relative_to(PROJECT_ROOT)} has shape {labels.shape}, expected ({N_EXPECTED},)")
        rows.append(
            {
                "arm": arm,
                "adjusted_rand_score": adjusted_rand_score(truth, labels),
                "fowlkes_mallows_score": fowlkes_mallows_score(truth, labels),
            }
        )

    table = pd.DataFrame(rows)
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(COMPARISON_CSV, index=False)
    print(f"[out] wrote {COMPARISON_CSV.relative_to(PROJECT_ROOT)}\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()

    write_json_atomic(
        MARKER,
        {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "arms": list(ARMS),
            "n_truth_groups": int(len(set(truth.tolist()))),
        },
    )
    print(f"[out] wrote {MARKER.relative_to(PROJECT_ROOT)}")
    print("[done] stage 5 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
