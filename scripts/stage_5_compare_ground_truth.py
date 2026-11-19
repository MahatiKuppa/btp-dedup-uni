#!/usr/bin/env python3
"""Stage 5 - compare the KMeans arms against the ground-truth group ids.

Run from the project root (needs stage 3 results for all four arms and stage 4):

    python3 scripts/stage_5_compare_ground_truth.py
    python3 scripts/stage_5_compare_ground_truth.py --force     # rerun despite the marker
    python3 scripts/stage_5_compare_ground_truth.py --cls colon_aca
    python3 scripts/stage_5_compare_ground_truth.py --check     # recompute, compare with stored csv, write nothing

For each arm in results[/<cls>]/kmeans_{uni,uni_pca128,resnet,resnet_pca128}/labels.npy computes
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
    N_EXPECTED,
    PROJECT_ROOT,
    ClassPaths,
    add_cls_arg,
    class_paths,
    die,
    ensure_dirs,
    write_json_atomic,
)

ARMS = ("kmeans_uni", "kmeans_uni_pca128", "kmeans_resnet", "kmeans_resnet_pca128")


def load_group_ids(P: ClassPaths) -> np.ndarray:
    """group_id per row, verified to follow embeddings/paths.txt row order."""
    if not P.group_map_csv.exists():
        die(f"missing {P.group_map_csv.relative_to(PROJECT_ROOT)}: run scripts/stage_4_match_groups.py --cls {P.cls}")
    mapping = pd.read_csv(P.group_map_csv)
    paths = [Path(line.strip()).name for line in (P.embeddings_dir / "paths.txt").read_text().splitlines() if line.strip()]
    if len(mapping) != N_EXPECTED or len(paths) != N_EXPECTED:
        die(f"expected {N_EXPECTED} rows, mapping={len(mapping)} paths.txt={len(paths)}")
    if mapping["my_filename"].tolist() != paths:
        die("my_filename order in the mapping does not match paths.txt")
    if mapping["group_id"].isna().any():
        die("mapping has missing group_id")
    print(f"[data] mapping aligned to embeddings/paths.txt ({len(mapping)} rows, "
          f"{mapping['group_id'].nunique()} distinct group_id)")
    return mapping["group_id"].to_numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cls_arg(parser)
    parser.add_argument("--force", action="store_true", help="rerun even if the completion marker exists")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compute the scores and compare them with the existing comparison csv; writes nothing",
    )
    args = parser.parse_args()

    P = class_paths(args.cls)
    COMPARISON_CSV, MARKER = P.comparison_csv, P.marker("stage_5_compare_ground_truth")

    if args.check:
        return check_existing(P)

    ensure_dirs(P)
    if not args.force and MARKER.exists() and COMPARISON_CSV.exists():
        print("already done (stage 5), skipping")
        return 0

    truth = load_group_ids(P)
    table = score_arms(P, truth)
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(COMPARISON_CSV, index=False)
    print(f"[out] wrote {COMPARISON_CSV.relative_to(PROJECT_ROOT)}\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()

    write_json_atomic(
        MARKER,
        {
            "class": P.cls,
            "completed": datetime.now().isoformat(timespec="seconds"),
            "arms": list(ARMS),
            "n_truth_groups": int(len(set(truth.tolist()))),
        },
    )
    print(f"[out] wrote {MARKER.relative_to(PROJECT_ROOT)}")
    print("[done] stage 5 complete")
    return 0


def score_arms(P: ClassPaths, truth: np.ndarray) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        path = P.results_dir / arm / "labels.npy"
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

    return pd.DataFrame(rows)


def check_existing(P: ClassPaths) -> int:
    """Recompute and compare with the stored csv, exact float equality. Writes nothing."""
    if not P.comparison_csv.exists():
        die(f"missing {P.comparison_csv.relative_to(PROJECT_ROOT)}")
    stored = pd.read_csv(P.comparison_csv, float_precision="round_trip")
    fresh = score_arms(P, load_group_ids(P))
    print(f"[check] {P.cls}: recomputed vs {P.comparison_csv.relative_to(PROJECT_ROOT)}")
    print(pd.concat([stored.set_index("arm").add_prefix("stored_"), fresh.set_index("arm").add_prefix("fresh_")], axis=1)
          .to_string(float_format=lambda v: f"{v:.10f}"))
    same = stored["arm"].tolist() == fresh["arm"].tolist() and all(
        np.array_equal(stored[col].to_numpy(), fresh[col].to_numpy())
        for col in ("adjusted_rand_score", "fowlkes_mallows_score")
    )
    if not same:
        # to_csv round-trips float64 exactly, so any difference here is real
        print("[check] MISMATCH")
        return 1
    print("[check] OK: identical to the stored values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
