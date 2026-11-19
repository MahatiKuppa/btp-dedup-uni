#!/usr/bin/env python3
"""Stage 4 - attach ground-truth group ids to the renamed colon_n tiles.

Run from the project root:

    python3 scripts/stage_4_match_groups.py
    python3 scripts/stage_4_match_groups.py --force     # rerun despite the marker
    python3 scripts/stage_4_match_groups.py --cls colon_aca

Hashes (sha256) every image in data/raw/colon_n (ours, colon_n_NNNN.jpeg) and
data/raw_tampapath/colon_n (original LC25000 names), builds a 1:1 mapping by hash,
then joins to data/lc25000_image_groups.csv on stem (filename minus extension).
Stops if any hash is unmatched or duplicated on either side.

Outputs:
    embeddings/my_filename_to_group_id.csv   my_filename, their_filename, group_id, local_cluster_label,
                                             their_filename_ambiguous (bool)
                                             (rows in sorted file order = embeddings/paths.txt)
    markers/stage_4_match_groups.done
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.project import (  # noqa: E402
    GROUPS_CSV,
    N_EXPECTED,
    PROJECT_ROOT,
    add_cls_arg,
    class_paths,
    die,
    ensure_dirs,
    list_images,
    sha256_of_file,
    write_json_atomic,
)

THEIR_SUFFIXES = (".jpeg", ".jpg")  # skips the *:Zone.Identifier files WSL leaves behind


def hash_folder(paths: list[Path], side: str) -> dict[str, list[str]]:
    """sha256 -> filenames (sorted). LC25000 itself holds byte-identical files, so a hash may repeat."""
    began = time.time()
    by_hash: dict[str, list[str]] = defaultdict(list)
    for index, path in enumerate(paths, 1):
        by_hash[sha256_of_file(path)].append(path.name)
        if index % 1000 == 0:
            print(f"[hash] {side}: {index}/{len(paths)}")
    repeated = [names for names in by_hash.values() if len(names) > 1]
    print(f"[hash] {side}: {len(paths)} files, {len(by_hash)} unique hashes in {time.time() - began:.1f}s | "
          f"{len(repeated)} hashes shared by {sum(map(len, repeated))} files")
    return by_hash


def their_images(original_dir: Path) -> list[Path]:
    return sorted(p for p in original_dir.iterdir() if p.is_file() and p.suffix.lower() in THEIR_SUFFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cls_arg(parser)
    parser.add_argument("--force", action="store_true", help="rerun even if the completion marker exists")
    args = parser.parse_args()

    P = class_paths(args.cls)
    RAW_DIR, ORIGINAL_DIR, GROUP_MAP_CSV, LABEL = P.raw_dir, P.original_dir, P.group_map_csv, P.cls
    MARKER = P.marker("stage_4_match_groups")

    ensure_dirs(P)
    if not args.force and MARKER.exists() and GROUP_MAP_CSV.exists():
        print("already done (stage 4), skipping")
        return 0

    for directory in (RAW_DIR, ORIGINAL_DIR):
        if not directory.is_dir():
            die(f"missing folder {directory}")
    if not GROUPS_CSV.exists():
        die(f"missing ground-truth CSV {GROUPS_CSV}")

    mine = list_images(RAW_DIR)
    theirs = their_images(ORIGINAL_DIR)
    print(f"[class] {LABEL}")
    print(f"[data] mine: {len(mine)} files | theirs: {len(theirs)} files (images only)")
    if len(mine) != N_EXPECTED or len(theirs) != N_EXPECTED:
        die(f"expected {N_EXPECTED} files on both sides, found mine={len(mine)} theirs={len(theirs)}")

    mine_by_hash = hash_folder(mine, "mine")
    theirs_by_hash = hash_folder(theirs, "theirs")

    only_mine = mine_by_hash.keys() - theirs_by_hash.keys()
    only_theirs = theirs_by_hash.keys() - mine_by_hash.keys()
    if only_mine or only_theirs:
        die(
            f"hash sets differ: {len(only_mine)} of mine have no match, {len(only_theirs)} of theirs unmatched\n"
            f"  e.g. mine: {[mine_by_hash[h][0] for h in list(only_mine)[:3]]}\n"
            f"  e.g. theirs: {[theirs_by_hash[h][0] for h in list(only_theirs)[:3]]}"
        )
    uneven = [h for h in mine_by_hash if len(mine_by_hash[h]) != len(theirs_by_hash[h])]
    if uneven:
        die(f"{len(uneven)} hashes have a different file count on each side, e.g. "
            f"{mine_by_hash[uneven[0]]} vs {theirs_by_hash[uneven[0]]}")
    print(f"[match] every hash matches with equal multiplicity ({len(mine_by_hash)} unique hashes)")

    # Byte-identical files cannot be told apart, so pair them by sorted position within the hash.
    their_for_mine: dict[str, str] = {}
    ambiguous: set[str] = set()
    for digest, my_names in mine_by_hash.items():
        for my_name, their_name in zip(sorted(my_names), sorted(theirs_by_hash[digest])):
            their_for_mine[my_name] = their_name
            if len(my_names) > 1:
                ambiguous.add(my_name)
    print(f"[match] {len(their_for_mine)} files mapped | {len(ambiguous)} ambiguous their_filename "
          f"(byte-identical duplicates; group_id unaffected if identical across the set)")

    mapping = pd.DataFrame(
        {
            "my_filename": [p.name for p in mine],  # sorted order = embeddings/paths.txt
            "their_filename": [their_for_mine[p.name] for p in mine],
            "their_filename_ambiguous": [p.name in ambiguous for p in mine],
        }
    )
    mapping["stem"] = mapping["their_filename"].map(lambda name: Path(name).stem)

    truth = pd.read_csv(GROUPS_CSV)
    truth = truth[truth["label"] == LABEL]
    if truth["stem"].duplicated().any():
        die(f"duplicate stems for label {LABEL} in {GROUPS_CSV.name}")
    print(f"[csv] {len(truth)} rows with label == {LABEL!r}")

    joined = mapping.merge(truth[["stem", "group_id", "local_cluster_label"]], on="stem", how="left", validate="1:1")
    missing = joined[joined["group_id"].isna()]
    if len(missing):
        die(f"{len(missing)} files have no group_id in the CSV, e.g. {missing['their_filename'].head(5).tolist()}")

    out = joined[
        ["my_filename", "their_filename", "group_id", "local_cluster_label", "their_filename_ambiguous"]
    ].astype(
        {"group_id": int, "local_cluster_label": int}
    )
    assert len(out) == N_EXPECTED, f"expected {N_EXPECTED} rows, got {len(out)}"
    assert out["group_id"].notna().all(), "missing group_id"
    assert out["my_filename"].is_unique and out["their_filename"].is_unique

    # Ambiguous pairing is only harmless if every duplicate set sits in one group.
    mixed = 0
    for digest, my_names in mine_by_hash.items():
        if len(my_names) > 1 and out.loc[out["my_filename"].isin(my_names), "group_id"].nunique() > 1:
            mixed += 1
    if mixed:
        die(f"{mixed} byte-identical duplicate sets span more than one group_id; pairing is not safe")
    print("[check] every byte-identical duplicate set lies in a single group_id")

    GROUP_MAP_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(GROUP_MAP_CSV, index=False)
    print(f"[out] wrote {GROUP_MAP_CSV.relative_to(PROJECT_ROOT)} {out.shape}")

    sizes = out["group_id"].value_counts()
    print(f"\n[groups] distinct group_id : {sizes.size}")
    print(f"[groups] group size        : min={int(sizes.min())} max={int(sizes.max())} mean={sizes.mean():.2f}")
    print(f"[groups] singleton groups  : {int((sizes == 1).sum())}")
    print(f"[groups] size histogram    : {dict(sorted(Counter(sizes.tolist()).items()))}")

    gid, lcl = out["group_id"], out["local_cluster_label"]
    offsets = set((gid - lcl).tolist())
    if (gid == lcl).all():
        relation = "IDENTICAL"
    elif len(offsets) == 1:
        relation = f"OFFSET by a constant (group_id - local_cluster_label = {offsets.pop()})"
    else:
        # one-to-one relabeling, or genuinely different partitions?
        pair_count = len(set(zip(gid.tolist(), lcl.tolist())))
        same_partition = pair_count == gid.nunique() == lcl.nunique()
        relation = "DIFFERENT labels" + (
            " (same partition, just a non-constant relabeling)" if same_partition else " and DIFFERENT partitions"
        )
    print(f"[groups] group_id vs local_cluster_label for {LABEL}: {relation}")
    print(f"[groups] group_id range {int(gid.min())}..{int(gid.max())}, local_cluster_label range {int(lcl.min())}..{int(lcl.max())}\n")

    write_json_atomic(
        MARKER,
        {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "class": LABEL,
            "n_rows": len(out),
            "n_ambiguous_their_filename": len(ambiguous),
            "n_groups": int(sizes.size),
            "group_size_min": int(sizes.min()),
            "group_size_max": int(sizes.max()),
            "group_size_mean": float(sizes.mean()),
            "n_singletons": int((sizes == 1).sum()),
            "group_id_vs_local_cluster_label": relation,
        },
    )
    print(f"[out] wrote {MARKER.relative_to(PROJECT_ROOT)}")
    print("[done] stage 4 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
