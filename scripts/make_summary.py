#!/usr/bin/env python3
"""Write results/<cls>/SUMMARY.md for a non-default class, next to colon_n.

Run from the project root after stages 2-6 for the class:

    python3 scripts/make_summary.py --cls colon_aca

Everything is read from existing outputs; nothing is recomputed except two checks that
guard colon_n: stage 5 --check (writes nothing) and the sha256 verification of
markers/colon_n_hashes_before.txt.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.project import MARKERS_DIR, PROJECT_ROOT, add_cls_arg, class_paths, sha256_of_file  # noqa: E402

ARMS = (("uni1_full", "uni"), ("uni1_pca128", "uni_pca128"), ("resnet50_full", "resnet"), ("resnet50_pca128", "resnet_pca128"))
HASH_BEFORE = MARKERS_DIR / "colon_n_hashes_before.txt"


def read_marker(path: Path) -> dict:
    return json.loads(path.read_text())


def group_stats(group_map: Path) -> dict:
    sizes = pd.read_csv(group_map)["group_id"].value_counts()
    return {"n_groups": int(sizes.size), "min": int(sizes.min()), "max": int(sizes.max()), "mean": float(sizes.mean())}


def arm_table(cls: str) -> pd.DataFrame:
    P = class_paths(cls)
    ari = pd.read_csv(P.comparison_csv, float_precision="round_trip").set_index("arm")
    rows = []
    for name, arm in ARMS:
        m = pd.read_csv(P.results_dir / f"kmeans_{arm}" / "metrics.csv").iloc[0]
        rows.append(
            {
                "arm": name,
                "size_min": int(m["size_min"]),
                "size_max": int(m["size_max"]),
                "size_mean": float(m["size_mean"]),
                "size_std": float(m["size_std"]),
                "separation_ratio": float(m["separation_ratio"]),
                "ari": float(ari.loc[f"kmeans_{arm}", "adjusted_rand_score"]),
            }
        )
    return pd.DataFrame(rows).set_index("arm")


def verify_colon_n_hashes() -> tuple[bool, str]:
    if not HASH_BEFORE.exists():
        return False, f"{HASH_BEFORE.relative_to(PROJECT_ROOT)} missing"
    total = bad = 0
    problems = []
    for line in HASH_BEFORE.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        total += 1
        path = PROJECT_ROOT / rel
        if not path.is_file() or sha256_of_file(path) != digest:
            bad += 1
            problems.append(rel)
    n_core = sum(1 for l in HASH_BEFORE.read_text().splitlines() if l.split(maxsplit=1)[1].startswith(("embeddings/", "results/")))
    verdict = f"{total - bad}/{total} files identical ({n_core} under embeddings/ and results/)"
    if problems:
        verdict += f"; DIFFERENT OR MISSING: {problems[:10]}"
    return not problems, verdict


def stage5_check_colon_n() -> tuple[bool, str]:
    run = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "stage_5_compare_ground_truth.py"), "--cls", "colon_n", "--check"],
        capture_output=True, text=True,
    )
    last = [l for l in run.stdout.splitlines() if l.startswith("[check]")][-1:]
    return run.returncode == 0, (last[0] if last else run.stderr.strip()[-200:])


def versions() -> dict:
    out = {"python": platform.python_version()}
    for label, module in (("torch", "torch"), ("torchvision", "torchvision"), ("timm", "timm"), ("numpy", "numpy"),
                          ("scikit-learn", "sklearn"), ("umap-learn", "umap"), ("scipy", "scipy"), ("pandas", "pandas"),
                          ("matplotlib", "matplotlib"), ("opencv", "cv2"), ("datasets", "datasets"),
                          ("huggingface_hub", "huggingface_hub")):
        try:
            out[label] = getattr(importlib.import_module(module), "__version__", "?")
        except ImportError:
            out[label] = "not installed"
    return out


def md_table(frame: pd.DataFrame, formats: dict) -> list[str]:
    cols = list(frame.columns)
    lines = ["| arm | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for arm, row in frame.iterrows():
        lines.append(f"| {arm} | " + " | ".join(formats[c].format(row[c]) for c in cols) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cls_arg(parser)
    args = parser.parse_args()
    P = class_paths(args.cls)
    N = class_paths("colon_n")

    stats, stats_n = group_stats(P.group_map_csv), group_stats(N.group_map_csv)
    st4, st4_n = read_marker(P.marker("stage_4_match_groups")), read_marker(N.marker("stage_4_match_groups"))
    st6 = read_marker(P.marker("stage_6_final_dedup"))
    n_images = len(pd.read_csv(P.group_map_csv))
    n_reps = len(list(P.final_dir.glob("*.jpeg")))
    runtimes = pd.read_csv(P.results_dir / "stage_runtimes.csv")
    table, table_n = arm_table(args.cls), arm_table("colon_n")
    st3 = read_marker(P.marker("stage_3_kmeans_uni"))
    st2 = read_marker(P.marker("stage_2_embed"))
    st2r = read_marker(P.marker("stage_2_embed_resnet"))

    ok5, msg5 = stage5_check_colon_n()
    okh, msgh = verify_colon_n_hashes()  # after the stage 5 check, so the check is covered too

    fmt = {"size_min": "{:.0f}", "size_max": "{:.0f}", "size_mean": "{:.2f}", "size_std": "{:.2f}", "separation_ratio": "{:.3f}", "ari": "{:.4f}"}
    ver = versions()

    lines = [
        f"# {args.cls} vs colon_n: dedup pipeline summary",
        "",
        "Same pipeline, same settings for both classes (UNI v1, ResNet50, KMeans k=250 random_state=42 n_init=10, PCA-128).",
        "The colon_n columns are read from the existing results, not recomputed.",
        "",
        "## Data and published groups",
        "",
        "| | " + args.cls + " | colon_n |",
        "|---|---|---|",
        f"| images | {n_images} | {len(pd.read_csv(N.group_map_csv))} |",
        f"| published groups | {stats['n_groups']} | {stats_n['n_groups']} |",
        f"| group size min / max / mean | {stats['min']} / {stats['max']} / {stats['mean']:.2f} | {stats_n['min']} / {stats_n['max']} / {stats_n['mean']:.2f} |",
        f"| singleton groups | {st4['n_singletons']} | {st4_n['n_singletons']} |",
        f"| files with ambiguous their_filename (byte-identical duplicates) | {st4['n_ambiguous_their_filename']} | {st4_n['n_ambiguous_their_filename']} |",
        f"| representatives in data/final/{args.cls} | {n_reps} | {len(list(N.final_dir.glob('*.jpeg')))} |",
        "",
        f"KMeans uses k=250 for both classes; {args.cls} has {stats['n_groups']} published groups and colon_n has {stats_n['n_groups']}, so ARI is scored against those.",
        "Duplicate handling is identical to colon_n: byte-identical files are paired by sorted position within the hash, and stage 4 stops unless every duplicate set lies in one group_id.",
        "",
        f"## {args.cls}: KMeans arms (k=250)",
        "",
        *md_table(table, fmt),
        "",
        "## colon_n: KMeans arms (k=250, existing results)",
        "",
        *md_table(table_n, fmt),
        "",
        "Cluster size columns are over the 250 KMeans clusters. ARI is against the published groups.",
        "",
        "## Stage runtimes (wall clock, " + args.cls + ")",
        "",
        "| stage | seconds | minutes | exit |",
        "|---|---|---|---|",
        *[f"| {r.stage} | {r.seconds} | {r.seconds / 60:.1f} | {r.exit_code} |" for r in runtimes.itertuples()],
        f"| total | {int(runtimes['seconds'].sum())} | {runtimes['seconds'].sum() / 60:.1f} | |",
        "",
        "The first stage_2_embed_uni row (exit 1) is the first attempt: it downloaded the HF dataset (~1.9 GB), then stopped on the label check because the dataset's name is 'adenocarcinomas', not 'adenocarcinoma'. The mapping was fixed and the rerun reused the cached dataset, so the later stage_2_embed_uni row has no download time; the download is that first row's time.",
        "Both stage 2 rows are kept. The total includes the failed attempt.",
        "",
        "## Package versions",
        "",
        *[f"- {k}: {v}" for k, v in ver.items()],
        f"- sharpness backend (stage 6): {st6['sharpness_backend']}",
        f"- stage 2 UNI marker: timm {st2['timm_version']}, torch {st2['torch_version']}",
        f"- stage 2 ResNet50 marker: torchvision {st2r['torchvision_version']}, torch {st2r['torch_version']}",
        f"- stage 3 marker: sklearn {st3['sklearn_version']}, umap {st3['umap_version']}",
        "",
        "## colon_n integrity check",
        "",
        f"- stage 5 `--check` on colon_n (recompute vs results/ground_truth_comparison.csv, no writes): {'PASS' if ok5 else 'FAIL'} - {msg5}",
        f"- sha256 of colon_n files vs markers/colon_n_hashes_before.txt: {'PASS' if okh else 'FAIL'} - {msgh}",
        "",
    ]
    out = P.results_dir / "SUMMARY.md"
    out.write_text("\n".join(lines))
    print(f"[out] wrote {out.relative_to(PROJECT_ROOT)}")
    return 0 if (ok5 and okh) else 1


if __name__ == "__main__":
    sys.exit(main())
