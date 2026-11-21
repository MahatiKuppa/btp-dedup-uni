#!/usr/bin/env python3
"""Score the old Colab approaches 1 and 2 (Lab-histogram based) against the published LC25000-clean groups.

Run with the dedicated venv (it needs opencv; the project venvs do not have it):

    data/approach12_drive/venv_a12/bin/python scripts/approach12_score.py <step>

Steps (each one writes only to data/approach12_drive/work/ except `score`, which writes the results/ files):
    map      re-saved Colab images vs data/raw originals: pixel statistics and order check
    stage3   Lab features + KMeans(250) exactly as the notebooks, checked against the logs
    sub      approach 1 stage 4: sub-KMeans of the oversized clusters, checked against splits.csv;
             then the logged merges joined with union-find -> approach 1 partitions
    replay   approach 1 stage 5 ORB merge pass replayed from the logged representative files (check only)
    score    ARI / FMI of every label set against the published groups; writes results/approach12_*

KMeans steps (stage3, sub, replay) run in an environment with sklearn 1.7.2, which reproduces Colab's KMeans
bit for bit; 1.8.0 and 1.9.1 do not (see results/approach12_NOTES.md). `score` runs in the sklearn 1.9.1 venv.

Inputs live in data/approach12_drive/ (copied from the Colab Drive; see results/approach12_NOTES.md).
The notebook code reproduced here is copied from
  approach1_LC25000_colon_dedup_pipeline_final.ipynb  (compute_lab_histogram, sharpness_score, orb_*, stage 3/4/5)
  approach2_LC25000_colon_n_partition_refinement.ipynb (compute_variance_metrics)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "data" / "approach12_drive"
WORK = A / "work"
RAW_RESAVED = A / "raw_resaved"
LOG1, LOG2 = A / "approach1" / "logs", A / "approach2" / "logs"
CLASSES = ("colon_n", "colon_aca")
N = 5000
K = 250
EXPECTED_SZ = N / K
SPLIT_THRESHOLD = 1.5
MERGE_ORB_THRESHOLD = 40
N_NEIGHBOURS = 10


def die(msg: str) -> None:
    print(f"[STOP] {msg}", file=sys.stderr)
    sys.exit(2)


def resaved_path(cls: str, i: int) -> Path:
    return RAW_RESAVED / cls / f"{cls}_{i:05d}.jpeg"


def orig_path(cls: str, i: int) -> Path:
    return ROOT / "data" / "raw" / cls / f"{cls}_{i:04d}.jpeg"


# --------------------------------------------------------------------------- #
# notebook functions, verbatim logic
# --------------------------------------------------------------------------- #
def compute_lab_histogram(image_path, bins=32):
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    parts = []
    for ch in range(3):
        h = cv2.calcHist([lab], [ch], None, [bins], [0, 256])
        parts.append(cv2.normalize(h, h).flatten())
    return np.concatenate(parts)


def sharpness_score(image_path):
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def compute_variance_metrics(features, labels, centers=None):
    """approach 2, cell 7: intra, inter, separation ratio."""
    if centers is None:
        centers = np.array([features[labels == k].mean(axis=0) if (labels == k).any()
                            else np.zeros(features.shape[1])
                            for k in range(labels.max() + 1)])
    intra = np.mean([np.sum((features[i] - centers[labels[i]]) ** 2) for i in range(len(features))])
    non_empty = [k for k in range(len(centers)) if (labels == k).any()]
    if len(non_empty) < 2:
        inter = 0.0
    else:
        c = centers[non_empty]
        d2 = np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=2)
        iu = np.triu_indices(len(c), k=1)
        inter = float(d2[iu].mean())
    return {"intra": float(intra), "inter": float(inter),
            "ratio": float(inter / intra) if intra > 0 else 0.0,
            "n_clusters_nonempty": len(non_empty)}


# --------------------------------------------------------------------------- #
# step 1: mapping
# --------------------------------------------------------------------------- #
def thumb(arr: np.ndarray) -> np.ndarray:
    h, w, c = arr.shape
    assert h % 32 == 0 and w % 32 == 0, arr.shape
    return arr.reshape(32, h // 32, 32, w // 32, c).mean(axis=(1, 3)).astype(np.float32).ravel()


def step_map() -> None:
    from PIL import Image

    WORK.mkdir(exist_ok=True)
    rows = []
    for cls in CLASSES:
        print(f"[map] {cls}")
        ta, tb, h_orig, rec = [], [], [], []
        for i in range(N):
            ra, oa = resaved_path(cls, i), orig_path(cls, i)
            with Image.open(ra) as im:
                a = np.asarray(im.convert("RGB"))
            with Image.open(oa) as im:
                b = np.asarray(im.convert("RGB"))
            if a.shape != b.shape:
                die(f"{cls} {i}: shape resaved {a.shape} vs original {b.shape}")
            d = np.abs(a.astype(np.int16) - b.astype(np.int16))
            rec.append((float(d.mean()), int(d.max())))
            ta.append(thumb(a))
            tb.append(thumb(b))
            h_orig.append(hashlib.sha256(oa.read_bytes()).hexdigest())
            if i % 1000 == 999:
                print(f"  {i + 1}/{N}")
        ta, tb = np.array(ta), np.array(tb)
        from scipy.spatial.distance import cdist

        dist = cdist(ta, tb, metric="cityblock") / ta.shape[1]  # mean abs diff per thumbnail value
        hashes = np.array(h_orig)
        same = hashes[:, None] == hashes[None, :]  # includes the diagonal
        masked = np.where(same, np.inf, dist)  # other originals, byte-identical copies of i excluded
        best_j = masked.argmin(axis=1)
        for i in range(N):
            rows.append({"cls": cls, "i": i, "resaved": resaved_path(cls, i).name, "original": orig_path(cls, i).name,
                         "mean_abs_diff": rec[i][0], "max_abs_diff": rec[i][1],
                         "thumb_d_own": float(dist[i, i]), "thumb_d_best_other": float(masked[i, best_j[i]]),
                         "best_other_j": int(best_j[i]), "n_byte_identical_originals": int(same[i].sum() - 1)})
    out = pd.DataFrame(rows)
    out.to_csv(WORK / "mapping_stats.csv", index=False)
    for cls in CLASSES:
        p = out[out.cls == cls]
        nearest_is_own = (p.thumb_d_own < p.thumb_d_best_other).all()
        print(f"[map] {cls}: mean_abs_diff mean {p.mean_abs_diff.mean():.3f} p99 {p.mean_abs_diff.quantile(.99):.3f} max {p.mean_abs_diff.max():.3f} | "
              f"max_abs_diff max {p.max_abs_diff.max()} | thumb own max {p.thumb_d_own.max():.3f} vs best-other min {p.thumb_d_best_other.min():.3f} | own nearest for all: {nearest_is_own}")


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
REP_TOL = 1e-12  # logged representative accepted if it is a cluster member with sharpness within this relative gap of the max
_SHA: dict = {}


def sha_orig(cls: str, i: int) -> str:
    key = (cls, i)
    if key not in _SHA:
        _SHA[key] = hashlib.sha256(orig_path(cls, i).read_bytes()).hexdigest()
    return _SHA[key]


def ambiguous_flag(cls: str) -> np.ndarray:
    d = pd.read_csv(mapping_csv(cls))
    return d["their_filename_ambiguous"].to_numpy()


def mapping_csv(cls: str) -> Path:
    return ROOT / "embeddings" / ("" if cls == "colon_n" else cls) / "my_filename_to_group_id.csv"


def files_sorted(cls: str) -> list[Path]:
    import os

    d = RAW_RESAVED / cls
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpeg"))
    assert len(files) == N, (cls, len(files))
    return [d / f for f in files]


def rep_index(path: str) -> int:
    return int(Path(path).stem.split("_")[-1])


def kmeans(features: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(features)


def rep_check(cls: str, paths: list[Path], members: np.ndarray, logged_idx: int, amb: np.ndarray) -> dict:
    """Notebook rule: representative = argmax Laplacian variance (first on ties). Compare with the logged one."""
    sharp = np.array([sharpness_score(paths[i]) for i in members])
    mx = float(sharp.max())
    mine = int(members[int(sharp.argmax())])
    in_cluster = logged_idx in set(members.tolist())
    rel = float((mx - sharp[list(members).index(logged_idx)]) / mx) if in_cluster and mx > 0 else float("inf")
    r = {"logged": logged_idx, "mine": mine, "strict": mine == logged_idx, "logged_in_cluster": in_cluster,
         "rel_gap": rel, "tolerant": bool(in_cluster and rel <= REP_TOL), "n_tied_within_tol": int((sharp >= mx * (1 - REP_TOL)).sum())}
    if not r["strict"]:
        r["byte_identical_originals"] = bool(sha_orig(cls, mine) == sha_orig(cls, logged_idx))
        r["both_flagged_ambiguous_in_stage4_mapping"] = bool(amb[mine] and amb[logged_idx])
    return r


def summarize_rep_checks(checks: list[dict]) -> dict:
    mism = [c for c in checks if not c["strict"]]
    return {"n": len(checks), "strict_match": len(checks) - len(mism), "tolerant_match": sum(c["tolerant"] for c in checks),
            "strict_mismatches": len(mism), "max_rel_gap_of_mismatches": max([c["rel_gap"] for c in mism], default=0.0),
            "mismatches_byte_identical_originals": sum(c.get("byte_identical_originals", False) for c in mism),
            "mismatches_both_flagged_ambiguous": sum(c.get("both_flagged_ambiguous_in_stage4_mapping", False) for c in mism),
            "mismatch_details": mism}


def piece_name(cid: int, sid=None) -> str:
    return f"cluster{cid:04d}" if sid is None else f"cluster{cid:04d}_sub{sid}"


def versions() -> dict:
    out = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}
    for label, mod in (("sklearn", "sklearn"), ("opencv", "cv2"), ("scipy", "scipy"), ("pillow", "PIL")):
        try:
            out[label] = __import__(mod).__version__
        except ImportError:
            out[label] = "not installed"
    return out


# --------------------------------------------------------------------------- #
# step 2: stage 3 (features + KMeans) checked against the logs
# --------------------------------------------------------------------------- #
def step_stage3() -> None:
    from sklearn.metrics import adjusted_rand_score

    WORK.mkdir(exist_ok=True)
    report = {"versions": versions()}
    ok_all = True
    for cls in CLASSES:
        paths = files_sorted(cls)
        amb = ambiguous_flag(cls)
        print(f"[stage3] {cls}: Lab histograms")
        feats = np.array([compute_lab_histogram(p) for p in paths])
        assert feats.shape == (N, 96) and feats.dtype == np.float32, (feats.shape, feats.dtype)
        np.save(WORK / f"features_{cls}.npy", feats)
        print(f"[stage3] {cls}: KMeans k={K}")
        labels = kmeans(feats, K)
        np.save(WORK / f"labels_stage3_{cls}.npy", labels)
        r = {}
        ok = True
        if cls == "colon_n":
            saved_f = np.load(LOG2 / "colon_n_features.npy")
            saved_l = np.load(LOG2 / "colon_n_labels_baseline.npy")
            diff = np.abs(feats.astype(np.float64) - saved_f.astype(np.float64))
            r["features_vs_saved"] = {"max_abs_diff": float(diff.max()), "all_exactly_equal": bool(np.array_equal(feats, saved_f))}
            same = bool(np.array_equal(labels, saved_l))
            r["recomputed_labels_vs_labels_baseline"] = {"identical": same, "n_differ": int((labels != saved_l).sum()),
                                                        "ari_between": float(adjusted_rand_score(saved_l, labels))}
            ok &= same and r["features_vs_saved"]["all_exactly_equal"]

        cl = pd.read_csv(LOG1 / f"{cls}_clusters.csv")
        counts = np.bincount(labels, minlength=K)
        n_eq = bool((counts[cl.cluster_id.to_numpy()] == cl.n_members.to_numpy()).all())
        r["clusters_csv_n_members_equal_all_250"] = n_eq
        ok &= n_eq
        checks = []
        for row in cl.itertuples():
            members = np.where(labels == row.cluster_id)[0]
            c = rep_check(cls, paths, members, rep_index(row.representative), amb)
            c["cluster_id"] = int(row.cluster_id)
            checks.append(c)
        r["representatives"] = summarize_rep_checks(checks)
        ok &= r["representatives"]["tolerant_match"] == len(checks)
        r["pass"] = bool(ok)
        ok_all &= ok
        report[cls] = r
        rr = {k: v for k, v in r["representatives"].items() if k != "mismatch_details"}
        print(f"[stage3] {cls}: pass={ok} | n_members equal: {n_eq} | reps {rr}")
    (WORK / "stage3_report.json").write_text(json.dumps(report, indent=1))
    if not ok_all:
        die("stage3 checks failed, see work/stage3_report.json")


# --------------------------------------------------------------------------- #
# step 3: approach 1 stage 4 (sub-KMeans) + logged merges (union-find)
# --------------------------------------------------------------------------- #
def step_sub() -> None:
    WORK.mkdir(exist_ok=True)
    if not json.loads((WORK / "stage3_report.json").read_text()).get("colon_n", {}).get("pass"):
        die("stage3 report missing or failed")
    report = {"versions": versions()}
    ok_all = True
    for cls in CLASSES:
        labels = np.load(WORK / f"labels_stage3_{cls}.npy")
        feats = np.load(WORK / f"features_{cls}.npy")
        paths = files_sorted(cls)
        amb = ambiguous_flag(cls)
        cl = pd.read_csv(LOG1 / f"{cls}_clusters.csv")
        sp = pd.read_csv(LOG1 / f"{cls}_splits.csv")
        mg = pd.read_csv(LOG1 / f"{cls}_merges.csv")
        fm = pd.read_csv(LOG1 / "final_manifest.csv")
        fm = fm[fm["class"] == cls]
        r: dict = {}

        over = cl[cl.n_members > EXPECTED_SZ * SPLIT_THRESHOLD].copy()
        over["est_sub"] = (over.n_members / EXPECTED_SZ).round().clip(lower=2).astype(int)
        r["oversized_ids_equal_splits_csv"] = bool(set(over.cluster_id) == set(sp.orig_cluster_id))
        r["n_oversized"] = int(len(over))
        logged = {(int(x.orig_cluster_id), int(x.sub_id)): x for x in sp.itertuples()}
        r["n_sub_per_cluster_equal"] = bool(over.set_index("cluster_id").est_sub.sort_index().tolist()
                                            == sp.groupby("orig_cluster_id").size().sort_index().tolist())

        pieces = np.full(N, -1, dtype=np.int32)
        names: list[str] = []
        saved_as: dict[str, str] = {}
        checks, size_ok, keys_ok = [], True, True
        seen = set()
        for row in cl.itertuples():
            cid = int(row.cluster_id)
            members = np.where(labels == cid)[0]
            if cid not in set(over.cluster_id):
                pieces[members] = len(names)
                names.append(piece_name(cid))
                saved_as[piece_name(cid)] = row.saved_as
                continue
            n_sub = int(over.loc[over.cluster_id == cid, "est_sub"].iloc[0])
            sub_labels = kmeans(feats[members], min(n_sub, len(members)))
            for sid in range(sub_labels.max() + 1):
                sub = members[sub_labels == sid]
                if len(sub) == 0:
                    continue
                key = (cid, sid)
                seen.add(key)
                if key not in logged:
                    keys_ok = False
                    continue
                lg = logged[key]
                size_ok &= int(lg.n_members) == len(sub)
                c = rep_check(cls, paths, sub, rep_index(lg.representative), amb)
                c["piece"] = piece_name(cid, sid)
                checks.append(c)
                pieces[sub] = len(names)
                names.append(piece_name(cid, sid))
                saved_as[piece_name(cid, sid)] = lg.saved_as
        keys_ok &= seen == set(logged)
        assert (pieces >= 0).all()
        r["sub_keys_equal_logged"] = bool(keys_ok)
        r["sub_sizes_equal_logged"] = bool(size_ok)
        r["sub_representatives"] = summarize_rep_checks(checks)
        r["n_pieces"] = len(names)
        r["expected_n_pieces"] = K - len(over) + len(sp)

        # logged merges -> union-find over pieces
        name_of_saved = {v: k for k, v in saved_as.items()}
        parent = list(range(len(names)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        idx = {n: i for i, n in enumerate(names)}
        unknown = [n for n in list(mg.kept) + list(mg.removed) if n not in name_of_saved]
        r["merge_names_unknown"] = len(unknown)
        for kept, removed in zip(mg.kept, mg.removed):
            if kept in name_of_saved and removed in name_of_saved:
                a, b = find(idx[name_of_saved[kept]]), find(idx[name_of_saved[removed]])
                if a != b:
                    parent[b] = a
        roots = np.array([find(i) for i in range(len(names))])
        comp = {rt: j for j, rt in enumerate(sorted(set(roots.tolist())))}
        a1_final = np.array([comp[roots[p]] for p in pieces], dtype=np.int32)
        r["n_groups_after_merges"] = int(a1_final.max() + 1)
        r["n_groups_expected_from_manifest"] = int(len(fm))
        survivors = set(saved_as.values()) - set(mg.removed)
        r["survivors_equal_final_manifest"] = bool(survivors == set(fm.filename))
        r["n_merges"] = int(len(mg))
        r["merged_kept_later_removed"] = int(len(set(mg.kept) & set(mg.removed)))

        ok = (r["oversized_ids_equal_splits_csv"] and r["n_sub_per_cluster_equal"] and keys_ok and size_ok
              and r["sub_representatives"]["tolerant_match"] == len(checks) and r["n_pieces"] == r["expected_n_pieces"]
              and r["merge_names_unknown"] == 0 and r["n_groups_after_merges"] == len(fm) and r["survivors_equal_final_manifest"])
        r["pass"] = bool(ok)
        ok_all &= ok
        report[cls] = r
        if ok:
            np.save(WORK / f"labels_a1_after_split_{cls}.npy", pieces)
            np.save(WORK / f"labels_a1_final_{cls}.npy", a1_final)
            (WORK / f"pieces_{cls}.json").write_text(json.dumps({"names": names, "saved_as": saved_as}))
        rr = {k: v for k, v in r["sub_representatives"].items() if k != "mismatch_details"}
        print(f"[sub] {cls}: pass={ok} | pieces {r['n_pieces']} (expected {r['expected_n_pieces']}) | groups {r['n_groups_after_merges']} (manifest {len(fm)}) | "
              f"survivors==manifest {r['survivors_equal_final_manifest']} | sub reps {rr}")
    (WORK / "sub_report.json").write_text(json.dumps(report, indent=1))
    if not ok_all:
        die("sub / merge checks failed for at least one class; no a1 partition written for it, see work/sub_report.json")


# --------------------------------------------------------------------------- #
# step 4: replay of approach 1 stage 5 (ORB merges), check only
# --------------------------------------------------------------------------- #
def step_replay() -> None:
    import cv2

    report = {"versions": versions()}
    for cls in CLASSES:
        pieces_json = WORK / f"pieces_{cls}.json"
        if not pieces_json.exists():
            die(f"{cls}: no pieces file (sub step did not pass)")
        saved_as = json.loads(pieces_json.read_text())["saved_as"]
        cl = pd.read_csv(LOG1 / f"{cls}_clusters.csv")
        sp = pd.read_csv(LOG1 / f"{cls}_splits.csv")
        src = dict(zip(cl.saved_as, cl.representative))
        src.update(dict(zip(sp.saved_as, sp.representative)))
        # rep_dir at stage 5 = the logged representative file of every piece, sorted by file name
        rep_files = sorted(saved_as.values())
        rep_paths = [resaved_path(cls, rep_index(src[f])) for f in rep_files]
        feats = np.array([compute_lab_histogram(p) for p in rep_paths])

        orb = cv2.ORB_create(nfeatures=500)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        cache: dict = {}

        def desc(p):
            if p not in cache:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                cache[p] = None if img is None else orb.detectAndCompute(img, None)[1]
            return cache[p]

        def orb_score(pa, pb):
            da, db = desc(pa), desc(pb)
            if da is None or db is None:
                return 0
            return sum(1 for m in bf.match(da, db) if m.distance < 40)

        dists = np.linalg.norm(feats[:, None, :] - feats[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        k = min(N_NEIGHBOURS, len(rep_paths) - 1)
        pairs = {(min(i, j), max(i, j)) for i in range(len(rep_paths)) for j in np.argsort(dists[i])[:k]}
        merged, removed = [], set()
        for i, j in sorted(pairs):
            if i in removed or j in removed:
                continue
            score = orb_score(rep_paths[i], rep_paths[j])
            if score >= MERGE_ORB_THRESHOLD:
                si, sj = sharpness_score(rep_paths[i]), sharpness_score(rep_paths[j])
                drop, keep = (j, i) if si >= sj else (i, j)
                removed.add(drop)
                merged.append({"kept": rep_files[keep], "removed": rep_files[drop], "orb_score": int(score)})
        logged = pd.read_csv(LOG1 / f"{cls}_merges.csv")
        lg = [(a, b, int(c)) for a, b, c in zip(logged.kept, logged.removed, logged.orb_score)]
        rp = [(m["kept"], m["removed"], m["orb_score"]) for m in merged]
        pair_set = lambda x: {(a, b) for a, b, _ in x}
        r = {"n_pairs_checked_candidates": len(pairs), "n_replayed_merges": len(rp), "n_logged_merges": len(lg),
             "identical_sequence": rp == lg, "same_kept_removed_pairs": pair_set(rp) == pair_set(lg),
             "n_pairs_in_both": len(pair_set(rp) & pair_set(lg)),
             "only_in_replay": sorted(pair_set(rp) - pair_set(lg))[:10], "only_in_log": sorted(pair_set(lg) - pair_set(rp))[:10],
             "orb_scores_equal_for_common_pairs": all(dict(((a, b), c) for a, b, c in rp)[x] == dict(((a, b), c) for a, b, c in lg)[x]
                                                      for x in pair_set(rp) & pair_set(lg)),
             "n_representative_files": len(rep_files)}
        report[cls] = r
        print(f"[replay] {cls}: {json.dumps({k: v for k, v in r.items() if not k.startswith('only')})}")
    (WORK / "replay_report.json").write_text(json.dumps(report, indent=1))


# --------------------------------------------------------------------------- #
# step 5: scoring (run in the sklearn 1.9.1 venv) + results files
# --------------------------------------------------------------------------- #
RES = ROOT / "results"
OUT_CSV = RES / "approach12_vs_ground_truth.csv"
OUT_NOTES = RES / "approach12_NOTES.md"
OUT_MAP = RES / "approach12_mapping_stats.csv"
OUT_CHECKS = RES / "approach12_checks.json"
OUT_LABELS = RES / "approach12_labels"
SKLEARN_TRIALS = [  # scripts/approach12_sklearn_check.py in a fresh venv per version, 2026-11-21 14:40-14:42, numpy 2.5.5
    {"sklearn": "1.9.1", "identical": False, "n_differ": 4717, "ari_vs_saved": None},
    {"sklearn": "1.8.0", "identical": False, "n_differ": 4717, "ari_vs_saved": 0.823826},
    {"sklearn": "1.7.2", "identical": True, "n_differ": 0, "ari_vs_saved": 1.0},
    {"sklearn": "1.6.1", "identical": True, "n_differ": 0, "ari_vs_saved": 1.0},
]


def group_ids(cls: str) -> np.ndarray:
    d = pd.read_csv(mapping_csv(cls))
    if d["my_filename"].tolist() != [f"{cls}_{i:04d}.jpeg" for i in range(N)] or d["group_id"].isna().any():
        die(f"{cls}: mapping csv order or group_id problem")
    return d["group_id"].to_numpy()


def describe(labels: np.ndarray) -> dict:
    sizes = np.bincount(np.unique(labels, return_inverse=True)[1])
    return {"n_clusters": int(len(sizes)), "size_min": int(sizes.min()), "size_median": float(np.median(sizes)), "size_max": int(sizes.max())}


def step_score() -> None:
    import sklearn
    from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score

    for pth in (OUT_CSV, OUT_NOTES, OUT_MAP, OUT_CHECKS, OUT_LABELS):
        if pth.exists():
            die(f"{pth.relative_to(ROOT)} exists, not overwriting")
    reports = {n: json.loads((WORK / f"{n}_report.json").read_text()) for n in ("stage3", "sub", "replay")}
    for cls in CLASSES:
        if not (reports["stage3"][cls]["pass"] and reports["sub"][cls]["pass"]):
            die(f"{cls}: stage3 or sub checks did not pass; no a1 partition to score")
    old191 = json.loads((WORK / "sklearn1.9.1_stage3_report.json").read_text())
    SKLEARN_TRIALS[0]["ari_vs_saved"] = old191["colon_n"]["recomputed_labels_vs_labels_baseline"]["ari_between"]
    assert old191["colon_n"]["recomputed_labels_vs_labels_baseline"]["n_differ"] == 4717

    scoring_versions = versions()
    assert scoring_versions["sklearn"] == "1.9.1", scoring_versions
    sets: dict[str, list] = {c: [] for c in CLASSES}
    feats = {c: np.load(WORK / f"features_{c}.npy") for c in CLASSES}
    saved_base = np.load(LOG2 / "colon_n_labels_baseline.npy")
    sets["colon_n"] = [
        ("lab_kmeans250", "main", "read: approach2/logs/colon_n_labels_baseline.npy (what ran on Colab)", saved_base),
        ("a1_final", "main", "rebuilt: stage-3 clusters + recomputed sub-KMeans + logged merges (union-find)", np.load(WORK / "labels_a1_final_colon_n.npy")),
        ("a2_final", "main", "read: approach2/logs/colon_n_labels_final.npy", np.load(LOG2 / "colon_n_labels_final.npy")),
        ("a1_after_split", "extra", "rebuilt: stage-3 clusters + recomputed sub-KMeans, before merges", np.load(WORK / "labels_a1_after_split_colon_n.npy")),
        ("a2_refined", "extra", "read: approach2/logs/colon_n_labels_refined.npy (after stage 4, before normalising to 250)", np.load(LOG2 / "colon_n_labels_refined.npy")),
    ]
    sets["colon_aca"] = [
        ("lab_kmeans250", "main", "recomputed (sklearn 1.7.2); n_members of all 250 clusters equal clusters.csv", np.load(WORK / "labels_stage3_colon_aca.npy")),
        ("a1_final", "main", "rebuilt: stage-3 clusters + recomputed sub-KMeans + logged merges (union-find)", np.load(WORK / "labels_a1_final_colon_aca.npy")),
        ("a1_after_split", "extra", "rebuilt: stage-3 clusters + recomputed sub-KMeans, before merges", np.load(WORK / "labels_a1_after_split_colon_aca.npy")),
    ]
    # approach 2 stage 3 labels must equal the lab_kmeans250 recomputed (colon_n)
    assert np.array_equal(np.load(WORK / "labels_stage3_colon_n.npy"), saved_base)
    lf = sets["colon_n"][2][3]
    assert lf.shape == (N,) and lf.dtype == np.int64 and np.array_equal(np.unique(lf), np.arange(K))

    # scoring path self-check: approach 3 (UNI) labels must give the stored stage-5 values exactly
    selfcheck = {}
    for cls, lab_path, csv_path in (("colon_n", RES / "kmeans_uni" / "labels.npy", RES / "ground_truth_comparison.csv"),
                                    ("colon_aca", RES / "colon_aca" / "kmeans_uni" / "labels.npy", RES / "colon_aca" / "ground_truth_comparison.csv")):
        g = group_ids(cls)
        lab = np.load(lab_path)
        stored = pd.read_csv(csv_path, float_precision="round_trip").set_index("arm").loc["kmeans_uni"]
        ari, fmi = adjusted_rand_score(g, lab), fowlkes_mallows_score(g, lab)
        selfcheck[cls] = {"ari": ari, "ari_stored": float(stored["adjusted_rand_score"]), "fmi": fmi, "fmi_stored": float(stored["fowlkes_mallows_score"]),
                          "exactly_equal": bool(ari == stored["adjusted_rand_score"] and fmi == stored["fowlkes_mallows_score"])}
        assert selfcheck[cls]["exactly_equal"], (cls, selfcheck[cls])

    # logged metrics (approach 2) vs the notebook function on its own saved arrays
    logged_metrics = {}
    base = pd.read_csv(LOG2 / "colon_n_metrics_baseline.csv").iloc[0]
    comp = pd.read_csv(LOG2 / "colon_n_metrics_comparison.csv").set_index("stage")
    fs = np.load(LOG2 / "colon_n_features.npy")
    m_base_centers = compute_variance_metrics(fs, saved_base, np.load(LOG2 / "colon_n_centers_baseline.npy"))
    m_base_means = compute_variance_metrics(fs, saved_base)
    m_final = compute_variance_metrics(fs, lf, np.array([fs[lf == k].mean(axis=0) for k in range(K)]))
    logged_metrics = {
        "baseline_saved_centers": {"mine": m_base_centers["ratio"], "logged": float(base["ratio"]), "equal": bool(m_base_centers["ratio"] == float(base["ratio"]))},
        "baseline_group_mean_centers": {"mine": m_base_means["ratio"], "logged": float(base["ratio"]), "abs_diff": abs(m_base_means["ratio"] - float(base["ratio"]))},
        "final_group_mean_centers": {"mine": m_final["ratio"], "logged": float(comp.loc["final", "ratio"]), "equal": bool(m_final["ratio"] == float(comp.loc["final", "ratio"]))},
    }

    rows = []
    for cls in CLASSES:
        g = group_ids(cls)
        for name, kind, source, lab in sets[cls]:
            assert lab.shape == (N,)
            m = compute_variance_metrics(feats[cls], np.unique(lab, return_inverse=True)[1].astype(np.int64))
            rows.append({"cls": cls, "label_set": name, "kind": kind, "source": source, "n_images": N, **describe(lab),
                         "ari": adjusted_rand_score(g, lab), "fmi": fowlkes_mallows_score(g, lab), "n_published_groups": int(len(np.unique(g))),
                         "sep_ratio": m["ratio"], "intra": m["intra"], "inter": m["inter"],
                         "sep_ratio_definition": "approach 2 compute_variance_metrics, group-mean centres" + (" (applied here; notebook 1 defines no ratio)" if name.startswith("a1") else "")})
    df = pd.DataFrame(rows)

    # tie analysis: are the two tied images in the same published group?
    ties = {}
    for cls in CLASSES:
        g = group_ids(cls)
        details = reports["stage3"][cls]["representatives"]["mismatch_details"] + reports["sub"][cls]["sub_representatives"]["mismatch_details"]
        ties[cls] = {"n_strict_mismatches": len(details), "same_published_group": int(sum(g[d["logged"]] == g[d["mine"]] for d in details)),
                     "byte_identical_originals": int(sum(d["byte_identical_originals"] for d in details))}

    blobs = {}
    for cls in CLASSES:
        g = group_ids(cls)
        fin = dict((n, l) for n, _, _, l in sets[cls])["a1_final"]
        spl = dict((n, l) for n, _, _, l in sets[cls])["a1_after_split"]
        sizes = np.bincount(fin)
        big = int(sizes.argmax())
        mem = np.where(fin == big)[0]
        blobs[cls] = {"size": int(sizes[big]), "second_largest": int(np.sort(sizes)[-2]), "n_pieces": int(len(np.unique(spl[mem]))),
                      "n_published_groups": int(len(np.unique(g[mem])))}

    mapstats = pd.read_csv(WORK / "mapping_stats.csv")
    map_summary = {}
    for cls in CLASSES:
        p = mapstats[mapstats.cls == cls]
        map_summary[cls] = {"n_pairs": int(len(p)), "mean_abs_diff_mean": float(p.mean_abs_diff.mean()), "mean_abs_diff_p99": float(p.mean_abs_diff.quantile(.99)),
                            "mean_abs_diff_max": float(p.mean_abs_diff.max()), "max_abs_diff_max": int(p.max_abs_diff.max()),
                            "thumb_own_max": float(p.thumb_d_own.max()), "thumb_best_other_min": float(p.thumb_d_best_other.min()),
                            "own_is_nearest_for_all": bool((p.thumb_d_own < p.thumb_d_best_other).all())}

    ref3 = {cls: pd.read_csv(csv, float_precision="round_trip") for cls, csv in
            (("colon_n", RES / "ground_truth_comparison.csv"), ("colon_aca", RES / "colon_aca" / "ground_truth_comparison.csv"))}

    checks = {"scoring_versions": scoring_versions, "kmeans_step_versions": reports["stage3"]["versions"], "sklearn_trials": SKLEARN_TRIALS,
              "mapping": map_summary, "stage3": reports["stage3"], "sub": reports["sub"], "replay": reports["replay"],
              "ties": ties, "largest_a1_final_group": blobs, "scoring_selfcheck_approach3_uni": selfcheck, "approach2_logged_metrics_vs_recomputed": logged_metrics}

    OUT_LABELS.mkdir(parents=True)
    for cls in CLASSES:
        for name, kind, source, lab in sets[cls]:
            np.save(OUT_LABELS / f"{name}_{cls}.npy", lab)
    df.to_csv(OUT_CSV, index=False)
    mapstats.to_csv(OUT_MAP, index=False)
    OUT_CHECKS.write_text(json.dumps(checks, indent=1, default=str))
    OUT_NOTES.write_text(build_notes(df, checks, ref3))
    print(df[["cls", "label_set", "kind", "n_clusters", "size_min", "size_median", "size_max", "ari", "fmi", "sep_ratio"]].to_string(index=False))
    print(f"[out] wrote {OUT_CSV.relative_to(ROOT)}, {OUT_NOTES.relative_to(ROOT)}, {OUT_MAP.relative_to(ROOT)}, {OUT_CHECKS.relative_to(ROOT)}, {OUT_LABELS.relative_to(ROOT)}/")


def build_notes(df: pd.DataFrame, c: dict, ref3: dict) -> str:
    def rep(cls, which):
        r = c[which][cls]["representatives" if which == "stage3" else "sub_representatives"]
        return f"{r['strict_match']}/{r['n']} strict, {r['tolerant_match']}/{r['n']} within {REP_TOL:g}; strict mismatches {r['strict_mismatches']} (max relative gap {r['max_rel_gap_of_mismatches']:.1e})"

    L = []
    w = L.append
    w("# Approaches 1 and 2 (Colab, Lab histograms) scored against the published LC25000-clean groups")
    w("")
    w("Script: `scripts/approach12_score.py` (steps map, stage3, sub, replay, score). Inputs: `data/approach12_drive/` (gitignored), copied from the Colab Drive:")
    w("the two notebooks, `approach1/logs/`, `approach2/logs/`, and the re-saved images `colon_raw_all.zip` (both zips have identical content: names, sizes and CRCs of all 10000 entries).")
    w("Approach 2 ran on colon_n only, so it has no colon_aca rows.")
    w("")
    w("## Environment")
    w("")
    kv, sv = c["kmeans_step_versions"], c["scoring_versions"]
    w(f"- KMeans-dependent steps (stage3, sub, replay): sklearn {kv['sklearn']}, opencv {kv['opencv']}, numpy {kv['numpy']}, pandas {kv['pandas']}, scipy {kv['scipy']}, python {kv['python']}.")
    w(f"- ARI/FMI scoring and the separation ratio: sklearn {sv['sklearn']}, numpy {sv['numpy']}, pandas {sv['pandas']} (same sklearn as the approach 3 scores in stage 5). The map step ran in the 1.9.1 venv (opencv {sv['opencv']}).")
    w("- Colab's own versions are not known: the notebooks have no saved pip output. Colab ran an sklearn that reproduces its KMeans (see next section); its opencv is unknown.")
    w("")
    w("## Colab's KMeans is reproduced by sklearn 1.6.1 / 1.7.2, not by 1.8.0 / 1.9.1")
    w("")
    w("On approach 2's saved `colon_n_features.npy` (bit-identical to the features I recompute from the re-saved images), `KMeans(n_clusters=250, random_state=42, n_init=10)` was fitted in a fresh venv per version and compared with the saved `colon_n_labels_baseline.npy`:")
    w("")
    w("| sklearn | identical to saved labels | labels differing | ARI vs saved |")
    w("|---|---|---|---|")
    for t in c["sklearn_trials"]:
        w(f"| {t['sklearn']} | {'yes' if t['identical'] else 'no'} | {t['n_differ']} | {t['ari_vs_saved']:.4f} |")
    w("")
    w("Versions 1.9.1 and 1.8.0 give a different partition (ARI 0.824 against the saved labels), so every KMeans step here ran in 1.7.2. Tried: 1.8.0, 1.7.2, 1.6.1 (3 versions, all installed cleanly, about 2 minutes in total); the 1.9.1 result comes from the main venv. `scripts/approach12_sklearn_check.py` is the check. The scratch venvs were deleted except the 1.7.2 one that was extended with opencv, pandas and pillow.")
    w("")
    w("## Checks and outcomes")
    w("")
    m = c["mapping"]
    w("**1. File mapping (re-saved Colab file `i` = original `data/raw/<cls>/<cls>_{i:04d}.jpeg`): PASS for both classes.**")
    for cls in CLASSES:
        s = m[cls]
        w(f"- {cls}: {s['n_pairs']} pairs, mean abs pixel difference mean {s['mean_abs_diff_mean']:.3f}, p99 {s['mean_abs_diff_p99']:.3f}, max {s['mean_abs_diff_max']:.3f} (largest single-pixel difference {s['max_abs_diff_max']}). "
          f"On 32x32 thumbnails the worst matched pair is at {s['thumb_own_max']:.2f}; the closest other original (byte-identical copies of the image excluded) is at {s['thumb_best_other_min']:.2f}. Own original is the nearest match for all 5000: {s['own_is_nearest_for_all']}.")
    w("")
    w("**2. Stage 3 (Lab histograms + KMeans k=250), recomputed from the re-saved images with sklearn 1.7.2: PASS for both classes.**")
    s3 = c["stage3"]
    f = s3["colon_n"]["features_vs_saved"]
    w(f"- colon_n features vs saved `colon_n_features.npy`: exactly equal = {f['all_exactly_equal']} (max abs diff {f['max_abs_diff']}).")
    r = s3["colon_n"]["recomputed_labels_vs_labels_baseline"]
    w(f"- colon_n recomputed labels vs saved `labels_baseline.npy`: identical = {r['identical']} (labels differing {r['n_differ']}).")
    for cls in CLASSES:
        w(f"- {cls}: `n_members` of all 250 clusters equal `clusters.csv`: {s3[cls]['clusters_csv_n_members_equal_all_250']}. Representatives (sharpest member by Laplacian variance): {rep(cls, 'stage3')}.")
    w("")
    w("**Representative rule.** The logged representative is accepted if it is a member of the cluster and its sharpness equals the cluster maximum within 1e-12 relative. All logged representatives satisfy this. The strict check (my argmax equals the logged pick) fails for the clusters above, always on two members whose sharpness is equal to float rounding, so which one `argmax` returns can depend on the last bits (possibly the opencv version: mine is 5.1.0, Colab's is unknown).")
    for cls in CLASSES:
        t = c["ties"][cls]
        w(f"- {cls}: {t['n_strict_mismatches']} strict mismatches (stage 3 and sub-KMeans together): {t['byte_identical_originals']} of the tied pairs are byte-identical duplicates in the original release (sha256 of the `data/raw` originals; none is flagged `their_filename_ambiguous` in the stage 4 mapping either); {t['same_published_group']} of them lie in the same published group.")
    w("")
    w("**3. Approach 1 stage 4 (sub-KMeans of oversized clusters, n_members > 30), recomputed: PASS for both classes.**")
    sb = c["sub"]
    for cls in CLASSES:
        x = sb[cls]
        w(f"- {cls}: {x['n_oversized']} oversized clusters = the ids in `splits.csv`: {x['oversized_ids_equal_splits_csv']}; sub-cluster count per cluster equal: {x['n_sub_per_cluster_equal']}; (cluster, sub) keys equal: {x['sub_keys_equal_logged']}; sizes equal: {x['sub_sizes_equal_logged']}; "
          f"pieces {x['n_pieces']} (expected {x['expected_n_pieces']}); representatives: {rep(cls, 'sub')}.")
    w("")
    w("**4. Approach 1 final partition (logged merges joined with union-find over pieces): PASS for both classes.**")
    for cls in CLASSES:
        x = sb[cls]
        w(f"- {cls}: {x['n_merges']} logged merges ({x['merged_kept_later_removed']} where a kept image is itself removed later, so union-find is needed); {x['n_groups_after_merges']} groups = {x['n_groups_expected_from_manifest']} rows of `final_manifest.csv`; "
          f"surviving representatives (all pieces minus the removed names) equal `final_manifest.csv` as a set: {x['survivors_equal_final_manifest']}; unknown names in merges.csv: {x['merge_names_unknown']}.")
    for cls in CLASSES:
        b = c["largest_a1_final_group"][cls]
        w(f"- {cls}: the largest group of the rebuilt partition has {b['size']} images (next largest {b['second_largest']}); it joins {b['n_pieces']} pieces that cover {b['n_published_groups']} different published groups.")
    w("")
    w("**5. Approach 1 stage 5 (ORB merge pass) replayed as a check only, from the logged representative file of every piece (not the re-picked one): the scored partition always uses the logged merges.**")
    for cls in CLASSES:
        x = c["replay"][cls]
        w(f"- {cls}: {x['n_representative_files']} representative files, {x['n_pairs_checked_candidates']} candidate pairs; replayed merges {x['n_replayed_merges']} vs logged {x['n_logged_merges']}: identical sequence (order, kept, removed, ORB score) = {x['identical_sequence']}.")
    w("")
    w("**6. Approach 2 saved labels.**")
    w("- `colon_n_labels_final.npy`: int64 (5000,), values 0..249 contiguous, 250 distinct (40,128 bytes = 5000 x 8 + 128 header; the notebook builds it from Python ints, the other label files are int32).")
    lm = c["approach2_logged_metrics_vs_recomputed"]
    w(f"- Separation ratio recomputed with the notebook function on the saved arrays: baseline with the saved KMeans centres {lm['baseline_saved_centers']['mine']!r} vs logged {lm['baseline_saved_centers']['logged']!r} (equal: {lm['baseline_saved_centers']['equal']}); "
      f"baseline with group-mean centres differs from logged by {lm['baseline_group_mean_centers']['abs_diff']:.2e}; final {lm['final_group_mean_centers']['mine']!r} vs logged {lm['final_group_mean_centers']['logged']!r} (equal: {lm['final_group_mean_centers']['equal']}).")
    w("- The notebook's own result: separation ratio fell from 76.45 (baseline) to 64.58 (final), which is the opposite of its 'want up' criterion.")
    w("")
    w("**7. Scoring path self-check (approach 3, UNI labels from stage 3 through the same code):**")
    for cls in CLASSES:
        x = c["scoring_selfcheck_approach3_uni"][cls]
        w(f"- {cls}: ARI {x['ari']!r} / FMI {x['fmi']!r} = stored stage 5 values: {x['exactly_equal']}.")
    w("")
    w("## What was recomputed and what was read")
    w("")
    w("- Read from file: `colon_n` `lab_kmeans250` (= `labels_baseline.npy`, what ran), `a2_final` (`labels_final.npy`), `a2_refined` (`labels_refined.npy`); the logs `clusters/splits/merges/final_manifest.csv`; the approach 2 metrics, features and centres used for the check in point 6.")
    w("- Recomputed (sklearn 1.7.2, opencv above): Lab features of both classes, the stage-3 KMeans of both classes (colon_n identical to the saved labels, used as a check only; colon_aca used for scoring), the sub-KMeans of the oversized clusters, the ORB replay (check only).")
    w("- Rebuilt from logs: `a1_final` and `a1_after_split` (stage-3 clusters, oversized ones replaced by their recomputed pieces, then the logged merges joined by union-find).")
    w("- Approach 1 has no saved label array and no separation ratio in its notebook; approach 2 has no colon_aca run.")
    w("")
    w("## Results (ARI / FMI against the published groups; colon_n 257 groups, colon_aca 249)")
    w("")
    w("Rows marked `extra` were added on request and are not part of the main comparison. `sep_ratio` is approach 2's function (inter / intra with group-mean centres) applied to every set; notebook 1 defines no ratio, so for `a1_*` rows it is an extra measurement, not a notebook value.")
    w("")
    w("| class | label set | kind | clusters | size min / median / max | ARI | FMI | sep. ratio |")
    w("|---|---|---|---|---|---|---|---|")
    for r_ in df.itertuples():
        w(f"| {r_.cls} | {r_.label_set} | {r_.kind} | {r_.n_clusters} | {r_.size_min} / {r_.size_median:g} / {r_.size_max} | {r_.ari:.4f} | {r_.fmi:.4f} | {r_.sep_ratio:.2f} |")
    w("")
    w("Approach 3 (UNI/ResNet KMeans k=250, stage 5, read from the existing csv files, not recomputed) for reference:")
    w("")
    w("| class | arm | ARI | FMI |")
    w("|---|---|---|---|")
    for cls in CLASSES:
        for r_ in ref3[cls].itertuples():
            w(f"| {cls} | {r_.arm} | {r_.adjusted_rand_score:.4f} | {r_.fowlkes_mallows_score:.4f} |")
    w("")
    w("Files: `results/approach12_vs_ground_truth.csv`, `results/approach12_mapping_stats.csv` (per file pair), `results/approach12_checks.json` (all check details), `results/approach12_labels/*.npy` (every scored label array, row order = `embeddings/**/my_filename_to_group_id.csv`).")
    return "\n".join(L) + "\n"

STEPS = {"map": step_map, "stage3": step_stage3, "sub": step_sub, "replay": step_replay, "score": step_score}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
        die(f"usage: approach12_score.py {{{','.join(STEPS)}}}")
    STEPS[sys.argv[1]]()
