#!/usr/bin/env python3
"""Stage 6 - build the final dedup set: one sharpest image per ground-truth group.

Run from the project root (needs stage 4):

    python3 scripts/stage_6_final_dedup.py
    python3 scripts/stage_6_final_dedup.py --force     # rerun despite the marker
    python3 scripts/stage_6_final_dedup.py --cls colon_aca

Sharpness = variance of the 3x3 Laplacian of the grayscale image (OpenCV if installed,
otherwise PIL + numpy with the same kernel and reflect-101 borders). Ties go to the
lowest filename. Inputs are only read; representatives are copied.

Outputs:
    data/final/<cls>/*.jpeg               one representative per group_id
    data/final/<cls>_manifest.csv         my_filename, their_filename, group_id, group_size
    markers/stage_6_final_dedup.done
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.project import (  # noqa: E402
    N_EXPECTED,
    PROJECT_ROOT,
    add_cls_arg,
    class_paths,
    die,
    ensure_dirs,
    write_json_atomic,
)

try:
    import cv2
except ImportError:
    cv2 = None


def laplacian_variance(path: Path) -> float:
    if cv2 is not None:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            die(f"cannot read {path}")
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float64)
    padded = np.pad(gray, 1, mode="reflect")
    lap = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * gray
    return float(lap.var())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cls_arg(parser)
    parser.add_argument("--force", action="store_true", help="rerun even if the completion marker exists")
    args = parser.parse_args()

    P = class_paths(args.cls)
    RAW_DIR, GROUP_MAP_CSV = P.raw_dir, P.group_map_csv
    FINAL_DIR, FINAL_MANIFEST = P.final_dir, P.final_manifest
    MARKER = P.marker("stage_6_final_dedup")

    ensure_dirs(P)
    if not args.force and MARKER.exists() and FINAL_MANIFEST.exists():
        print("already done (stage 6), skipping")
        return 0

    if not GROUP_MAP_CSV.exists():
        die(f"missing {GROUP_MAP_CSV.relative_to(PROJECT_ROOT)}: run scripts/stage_4_match_groups.py")
    mapping = pd.read_csv(GROUP_MAP_CSV)
    if len(mapping) != N_EXPECTED or mapping["group_id"].isna().any():
        die("mapping must have 5000 rows and no missing group_id")

    stale = list(FINAL_DIR.glob("*")) if FINAL_DIR.exists() else []
    if stale and not args.force:
        die(f"{FINAL_DIR.relative_to(PROJECT_ROOT)} already has {len(stale)} files; use --force to replace them")
    for path in stale:
        if path.is_file():
            path.unlink()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[sharp] Laplacian variance via {'OpenCV' if cv2 is not None else 'PIL + numpy'} on {len(mapping)} images ...")
    mapping["sharpness"] = [laplacian_variance(RAW_DIR / name) for name in mapping["my_filename"]]
    mapping["group_size"] = mapping.groupby("group_id")["my_filename"].transform("size")

    # highest sharpness per group; ties to the lowest filename
    ranked = mapping.sort_values(["group_id", "sharpness", "my_filename"], ascending=[True, False, True])
    chosen = ranked.groupby("group_id", sort=True).head(1)
    manifest = chosen[["my_filename", "their_filename", "group_id", "group_size"]].reset_index(drop=True)

    for name in manifest["my_filename"]:
        shutil.copy2(RAW_DIR / name, FINAL_DIR / name)
    manifest.to_csv(FINAL_MANIFEST, index=False)

    n_files = len(list(FINAL_DIR.glob("*")))
    n_groups = mapping["group_id"].nunique()
    assert n_files == n_groups, f"{n_files} files copied but {n_groups} distinct group_id"
    assert len(manifest) == n_groups and manifest["group_id"].is_unique
    print(f"[out] copied {n_files} representatives to {FINAL_DIR.relative_to(PROJECT_ROOT)}")
    print(f"[out] wrote {FINAL_MANIFEST.relative_to(PROJECT_ROOT)} {manifest.shape}")
    print(f"[check] file count {n_files} == distinct group_id {n_groups}")

    write_json_atomic(
        MARKER,
        {
            "class": P.cls,
            "completed": datetime.now().isoformat(timespec="seconds"),
            "n_representatives": n_files,
            "n_groups": int(n_groups),
            "sharpness_backend": "opencv" if cv2 is not None else "pil+numpy",
        },
    )
    print(f"[out] wrote {MARKER.relative_to(PROJECT_ROOT)}")
    print("[done] stage 6 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
