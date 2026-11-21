#!/usr/bin/env python3
"""Leakage check - how much do plain random 80/20 splits put one group on both sides?

Run from the project root:

    python3 scripts/leakage_check.py

For each class (colon_n, colon_aca) the 5000 images are split 1000 times 80/20, twice:
  random  sklearn train_test_split on the image indices, random_state = split index, no stratification
  group   GroupShuffleSplit(n_splits=1, test_size=0.2, random_state = split index) on group_id
and per split two numbers are recorded:
  (a) % of groups with images on both sides (denominator: all groups of the class)
  (b) % of test images whose group also appears in train

Groups are the published LC25000-clean group_id values in data/lc25000_image_groups.csv.
GroupShuffleSplit takes test_size as a fraction of groups, so its test share of images is
only about 20%; the mean test share is reported for both splitters.

Output:
    results/leakage_check.csv    one row per class and splitter: mean and sd over the splits
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.project import CLASSES, GROUPS_CSV, N_EXPECTED, PROJECT_ROOT, die  # noqa: E402

N_SPLITS = 1000
TEST_SIZE = 0.2
OUT_CSV = PROJECT_ROOT / "results" / "leakage_check.csv"


def split_metrics(codes: np.ndarray, n_groups: int, train: np.ndarray, test: np.ndarray) -> tuple[float, float, float]:
    in_train = np.bincount(codes[train], minlength=n_groups) > 0
    in_test = np.bincount(codes[test], minlength=n_groups) > 0
    both = float((in_train & in_test).sum() / n_groups * 100)
    test_leaked = float(in_train[codes[test]].mean() * 100)
    return both, test_leaked, len(test) / len(codes) * 100


def random_splits(codes: np.ndarray, n_groups: int) -> np.ndarray:
    index = np.arange(len(codes))
    rows = []
    for seed in range(N_SPLITS):
        train, test = train_test_split(index, test_size=TEST_SIZE, random_state=seed)
        rows.append(split_metrics(codes, n_groups, train, test))
    return np.array(rows)


def group_splits(codes: np.ndarray, n_groups: int) -> np.ndarray:
    index = np.arange(len(codes))
    rows = []
    for seed in range(N_SPLITS):
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train, test = next(splitter.split(index, groups=codes))
        rows.append(split_metrics(codes, n_groups, train, test))
    return np.array(rows)


def main() -> int:
    if not GROUPS_CSV.exists():
        die(f"missing {GROUPS_CSV}")
    table = pd.read_csv(GROUPS_CSV)

    rows = []
    for cls in CLASSES:
        part = table[table["label"] == cls]
        if len(part) != N_EXPECTED or part["group_id"].isna().any():
            die(f"{cls}: expected {N_EXPECTED} rows with a group_id, found {len(part)}")
        codes, uniques = pd.factorize(part["group_id"].to_numpy())
        n_groups = len(uniques)

        for name, run in (("random", random_splits), ("group", group_splits)):
            m = run(codes, n_groups)
            rows.append(
                {
                    "cls": cls,
                    "splitter": name,
                    "n_images": len(codes),
                    "n_groups": n_groups,
                    "n_splits": N_SPLITS,
                    "pct_groups_both_sides_mean": m[:, 0].mean(),
                    "pct_groups_both_sides_sd": m[:, 0].std(ddof=1),
                    "pct_test_images_leaked_mean": m[:, 1].mean(),
                    "pct_test_images_leaked_sd": m[:, 1].std(ddof=1),
                    "pct_test_images_mean": m[:, 2].mean(),
                }
            )
            r = rows[-1]
            print(
                f"{cls:9s} {name:6s} groups={n_groups} | groups on both sides {r['pct_groups_both_sides_mean']:6.2f} ± "
                f"{r['pct_groups_both_sides_sd']:.2f} % | test images with their group in train "
                f"{r['pct_test_images_leaked_mean']:6.2f} ± {r['pct_test_images_leaked_sd']:.2f} % | test share {r['pct_test_images_mean']:.2f} %"
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"[out] wrote {OUT_CSV.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
