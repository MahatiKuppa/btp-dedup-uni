#!/usr/bin/env python3
"""Discussion tests: rotation sensitivity of Lab histograms vs UNI (Test A) and the ORB merges of approach 1 (Test B).

Steps and the environment each one runs in (run from the project root, always with -I):

    venv_sk_1.7.2 (cv2, sklearn 1.7.2):   python scripts/discussion_tests.py lab | pk | orb | report
    venv (timm 0.9.8, as stage 2):        HF_HUB_OFFLINE=1 python scripts/discussion_tests.py uni
    venv_repro (matplotlib):              python scripts/discussion_tests.py plot

    lab     selection (seed 0), Lab histograms of the 100 images per class under every condition, Lab reference of all 5000
    pk      P@1 / P@5 in Lab space (upstream definition, copied) + validation of the copy on the stored UNI features
    uni     UNI embeddings (approach 3 transform and resize-only transform) of the same 100 x conditions
    orb     Test B: logged ORB merges vs published groups, thresholds, post-hoc ARI, real replay at each threshold (extra)
    report  testA.csv, testB.csv, NOTES.md under results/discussion_tests/
    plot    results/discussion_tests/testA_rotation.png

Intermediate files: data/approach12_drive/work/discussion/ (gitignored). Outputs are never overwritten.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "data" / "approach12_drive"
WORK = A / "work" / "discussion"
A12W = A / "work"
OUTD = ROOT / "results" / "discussion_tests"
UP = ROOT / "third_party" / "LC25000-clean"
CLASSES = ("colon_n", "colon_aca")
N = 5000
SIDE = 768
N_SEL = 100
ANGLES = (0, 5, 10, 15, 20, 25)
CONDS = [("rot", a) for a in ANGLES] + [("zoom", a) for a in ANGLES[1:]] + [("hflip", 0)]
THRESHOLDS = (40, 50, 60, 80, 100, 150)


def die(msg: str) -> None:
    print(f"[STOP] {msg}", file=sys.stderr)
    sys.exit(2)


def cname(c) -> str:
    return f"{c[0]}{c[1]}"


def mapping_csv(cls: str) -> Path:
    return ROOT / "embeddings" / ("" if cls == "colon_n" else cls) / "my_filename_to_group_id.csv"


def load_mapping(cls: str):
    import pandas as pd

    d = pd.read_csv(mapping_csv(cls))
    assert d["my_filename"].tolist() == [f"{cls}_{i:04d}.jpeg" for i in range(N)]
    return d


def orig_path(cls: str, i: int) -> Path:
    return ROOT / "data" / "raw" / cls / f"{cls}_{i:04d}.jpeg"


# --------------------------------------------------------------------------- #
# conditions
# --------------------------------------------------------------------------- #
def crop_side(angle: int) -> int:
    """Largest axis-aligned square without empty corners after rotating a 768 square by `angle`, minus a 2 px margin."""
    if angle == 0:
        return SIDE
    th = np.deg2rad(angle)
    s = int(np.floor(SIDE / (np.cos(th) + np.sin(th)))) - 2
    return s - s % 2


def apply_cond(im, cond):
    from PIL import Image

    kind, a = cond
    if kind == "hflip":
        return im.transpose(Image.FLIP_LEFT_RIGHT)
    s = crop_side(a)
    lo = (SIDE - s) // 2
    box = (lo, lo, lo + s, lo + s)
    src = im.rotate(a, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0)) if (kind == "rot" and a) else im
    out = src.crop(box)
    return out if s == SIDE else out.resize((SIDE, SIDE), Image.BILINEAR)


def selection(cls: str) -> dict:
    g = load_mapping(cls)["group_id"].to_numpy()
    rng = np.random.RandomState(0)
    chosen = rng.choice(np.unique(g), N_SEL, replace=False)
    idx = [int(rng.choice(np.where(g == gr)[0])) for gr in chosen]
    return {"groups": [int(x) for x in chosen], "index": idx}


def lab_from_rgb(arr: np.ndarray) -> np.ndarray:
    """approach 1 compute_lab_histogram on an in-memory RGB array."""
    import cv2

    img = np.ascontiguousarray(arr[:, :, ::-1])
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    parts = []
    for ch in range(3):
        h = cv2.calcHist([lab], [ch], None, [32], [0, 256])
        parts.append(cv2.normalize(h, h).flatten())
    return np.concatenate(parts)


# --------------------------------------------------------------------------- #
# step lab
# --------------------------------------------------------------------------- #
def step_lab() -> None:
    import cv2
    from PIL import Image

    WORK.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        sel = selection(cls)
        (WORK / f"{cls}_selection.json").write_text(json.dumps({**sel, "crop_side": {str(a): crop_side(a) for a in ANGLES}}))
        print(f"[lab] {cls}: reference Lab of {N} originals (PIL decode)")
        ref = np.array([lab_from_rgb(np.asarray(Image.open(orig_path(cls, i)).convert("RGB"))) for i in range(N)])
        np.save(WORK / f"{cls}_lab_ref.npy", ref)
        dec = max(float(np.abs(ref[i] - _cv2_lab(orig_path(cls, i))).max()) for i in sel["index"])
        prev = np.load(A12W / f"features_{cls}.npy")  # approach 1 features (re-saved images)
        feats = {}
        for cond in CONDS:
            feats[cname(cond)] = np.array([lab_from_rgb(np.asarray(apply_cond(Image.open(orig_path(cls, i)).convert("RGB"), cond))) for i in sel["index"]])
        np.savez(WORK / f"{cls}_lab_conds.npz", **feats)
        info = {"pil_vs_cv2_decode_max_abs_diff_selected": dec,
                "originals_vs_resaved_features_mean_abs_diff": float(np.abs(ref - prev).mean()),
                "originals_vs_resaved_features_max_abs_diff": float(np.abs(ref - prev).max()),
                "crop_side": {str(a): crop_side(a) for a in ANGLES}}
        (WORK / f"{cls}_lab_info.json").write_text(json.dumps(info))
        print(f"[lab] {cls}: {info}")


def _cv2_lab(path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    parts = []
    for ch in range(3):
        h = cv2.calcHist([lab], [ch], None, [32], [0, 256])
        parts.append(cv2.normalize(h, h).flatten())
    return np.concatenate(parts)


# --------------------------------------------------------------------------- #
# step pk: upstream precision@1 / precision@5 (copied verbatim from
# third_party/LC25000-clean/source/eval_utils.py, lines 223-271; that module imports umap, not available here)
# --------------------------------------------------------------------------- #
def is_closest_connected(i, dist_matrix, true_connectivity_matrix):
    dist_matrix[i, i] = np.inf
    closest_img_index = np.argmin(dist_matrix[i])
    return true_connectivity_matrix[i, closest_img_index]


def precision_at_1(features, true_connectivity_matrix, metric="euclidean"):
    from sklearn.metrics import pairwise_distances

    dist_matrix = pairwise_distances(features, features, metric=metric)
    num_total_images = features.shape[0]
    connected_to_closest_nbr_arr = np.zeros(num_total_images, dtype=int)
    for i in range(num_total_images):
        connected_to_closest_nbr_arr[i] = is_closest_connected(i, dist_matrix, true_connectivity_matrix)
    return np.mean(connected_to_closest_nbr_arr)


def get_closest_k_connection_labels(i, dist_matrix, true_connectivity_matrix, k=5):
    dist_matrix[i, i] = np.inf
    closest_img_indices_unsorted = np.argpartition(dist_matrix[i], k)[:k]
    closest_nbrs_sorted = closest_img_indices_unsorted[np.argsort(dist_matrix[i][closest_img_indices_unsorted])]
    return true_connectivity_matrix[i, closest_nbrs_sorted]


def precision_at_k(features, true_connectivity_matrix, k=5, metric="euclidean"):
    from sklearn.metrics import pairwise_distances

    dist_matrix = pairwise_distances(features, features, metric=metric)
    num_total_images = features.shape[0]
    connected_to_closest_k_nbrs_arr = np.zeros((num_total_images, k), dtype=int)
    for i in range(num_total_images):
        connected_to_closest_k_nbrs_arr[i] = get_closest_k_connection_labels(i, dist_matrix, true_connectivity_matrix, k)
    return np.mean(connected_to_closest_k_nbrs_arr, axis=(0, 1))


def connectivity(groups: np.ndarray) -> np.ndarray:
    m = (groups[:, None] == groups[None, :]).astype(int)
    np.fill_diagonal(m, 0)
    return m


def upstream_features(cls: str, norm: str):
    """third_party UNI features and the published group of each row, in upstream row order."""
    import pandas as pd

    d = UP / "feature_vectors" / cls / "UNI" / norm
    feats = np.load(d / "features.npy")
    ids = json.loads((d / "ids_2_img_paths.json").read_text())
    names = [Path(ids[str(i)]).name for i in range(len(ids))]
    pub = pd.read_csv(ROOT / "data" / "lc25000_image_groups.csv").set_index("filename")["group_id"]
    return feats, names, pub.loc[names].to_numpy()


def stored_pk(cls: str) -> dict:
    import pandas as pd

    out = {}
    theirs = pd.read_csv(next((UP / "eval_results").glob(f"*{cls}#*.csv")) if False else UP / "eval_results" / f"cancer_type={cls}#extractor_name=all#img_norm=all#distance_metric=all#dimensionality_reduction=all#clustering=all.csv").set_index("Unnamed: 0")
    mine = pd.read_csv(ROOT / "repro" / "out" / ("" if cls == "colon_n" else cls) / "phase3" / "eval_mine_11combos.csv").set_index("Unnamed: 0")
    for norm in ("imagenet", "resize_only"):
        key = f"UNI#{norm}#euclidean#NoReduction#kmeans"
        out[norm] = {"theirs": (float(theirs.loc[key, "precision@1"]), float(theirs.loc[key, "precision@5"])),
                     "mine": (float(mine.loc[key, "precision@1"]), float(mine.loc[key, "precision@5"]))}
    return out


def step_pk() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    res = {}
    for cls in CLASSES:
        stored = stored_pk(cls)
        r = {"stored": stored, "validation": {}}
        # validation: copied functions on the stored UNI features must reproduce my stored 11-combo values
        for norm in ("imagenet", "resize_only"):
            feats, names, grp = upstream_features(cls, norm)
            conn = connectivity(grp)
            p1, p5 = precision_at_1(feats, conn), precision_at_k(feats, conn, k=5)
            m1, m5 = stored[norm]["mine"]
            r["validation"][norm] = {"p1": float(p1), "p5": float(p5), "stored_mine_p1": m1, "stored_mine_p5": m5,
                                     "abs_diff_p1": abs(float(p1) - m1), "abs_diff_p5": abs(float(p5) - m5)}
            print(f"[pk] {cls} validation {norm}: P@1 {p1:.5f} (stored {m1:.5f}) P@5 {p5:.5f} (stored {m5:.5f})")
            if abs(float(p1) - m1) > 1e-3 or abs(float(p5) - m5) > 1e-3:
                die(f"{cls} {norm}: copied P@k does not reproduce the stored value")
        g = load_mapping(cls)["group_id"].to_numpy()
        lab = np.load(A12W / f"features_{cls}.npy")
        conn = connectivity(g)
        r["lab"] = {"p1": float(precision_at_1(lab, conn)), "p5": float(precision_at_k(lab, conn, k=5))}
        print(f"[pk] {cls} Lab histograms: P@1 {r['lab']['p1']:.5f} P@5 {r['lab']['p5']:.5f}")
        res[cls] = r
    (WORK / "pk.json").write_text(json.dumps(res, indent=1))


# --------------------------------------------------------------------------- #
# step uni
# --------------------------------------------------------------------------- #
def step_uni() -> None:
    import timm
    import torch
    from PIL import Image
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from torchvision.transforms import v2

    (WORK / "uni").mkdir(parents=True, exist_ok=True)
    model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True).eval()
    config = resolve_data_config(model.pretrained_cfg, model=model)
    tf_a3 = create_transform(**config)  # approach 3 / stage 2 transform: resize, centre crop, ImageNet normalisation
    tf_resize = v2.Compose([v2.Resize(224), v2.ToDtype(torch.float32, scale=True)])  # upstream resize_only on a uint8 tensor
    print(f"[uni] timm {timm.__version__} torch {torch.__version__} threads {torch.get_num_threads()} config {config}", flush=True)

    def embed(tensors):
        out = []
        with torch.inference_mode():
            for s in range(0, len(tensors), 20):
                out.append(model(torch.stack(tensors[s:s + 20])).numpy())
        return np.concatenate(out)

    for cls in CLASSES:
        sel = json.loads((WORK / f"{cls}_selection.json").read_text())
        for cond in CONDS:
            path = WORK / "uni" / f"{cls}_{cname(cond)}.npz"
            if path.exists():
                continue
            t0 = time.time()
            imgs = [apply_cond(Image.open(orig_path(cls, i)).convert("RGB"), cond) for i in sel["index"]]
            a = embed([tf_a3(im) for im in imgs])
            b = embed([tf_resize(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)) for im in imgs])
            tmp = path.with_name(path.name + ".tmp.npz")
            np.savez(tmp, imagenet=a, resize_only=b)
            tmp.rename(path)
            print(f"[uni] {cls} {cname(cond)} done in {time.time() - t0:.0f}s", flush=True)
    print("[uni] ALL DONE", flush=True)


# --------------------------------------------------------------------------- #
# step orb: Test B
# --------------------------------------------------------------------------- #
LOG1 = A / "approach1" / "logs"


def rep_index(path: str) -> int:
    return int(Path(path).stem.split("_")[-1])


def join_pieces(pieces: np.ndarray, names: list, saved_as: dict, merge_pairs) -> np.ndarray:
    """Union-find over pieces; merge_pairs = [(kept saved_as, removed saved_as)]. Same rebuild as approach12_score.py."""
    inv = {v: k for k, v in saved_as.items()}
    idx = {n: i for i, n in enumerate(names)}
    parent = list(range(len(names)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for kept, removed in merge_pairs:
        a, b = find(idx[inv[kept]]), find(idx[inv[removed]])
        if a != b:
            parent[b] = a
    roots = np.array([find(i) for i in range(len(names))])
    comp = {rt: j for j, rt in enumerate(sorted(set(roots.tolist())))}
    return np.array([comp[roots[p]] for p in pieces], dtype=np.int32)


def step_orb() -> None:
    import cv2
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score

    cv2.setNumThreads(2)
    WORK.mkdir(parents=True, exist_ok=True)
    out = {}
    for cls in CLASSES:
        g = load_mapping(cls)["group_id"].to_numpy()
        pj = json.loads((A12W / f"pieces_{cls}.json").read_text())
        names, saved_as = pj["names"], pj["saved_as"]
        cl = pd.read_csv(LOG1 / f"{cls}_clusters.csv")
        sp = pd.read_csv(LOG1 / f"{cls}_splits.csv")
        src = dict(zip(cl.saved_as, cl.representative))
        src.update(dict(zip(sp.saved_as, sp.representative)))
        rep = {sa: rep_index(src[sa]) for sa in saved_as.values()}
        pieces = np.load(A12W / f"labels_a1_after_split_{cls}.npy")
        final_ref = np.load(ROOT / "results" / "approach12_labels" / f"a1_final_{cls}.npy")
        mg = pd.read_csv(LOG1 / f"{cls}_merges.csv")
        merges = [{"kept": k, "removed": r, "orb_score": int(sc), "group_kept": int(g[rep[k]]), "group_removed": int(g[rep[r]]),
                   "same_group": bool(g[rep[k]] == g[rep[r]]), "rep_kept": rep[k], "rep_removed": rep[r]}
                  for k, r, sc in zip(mg.kept, mg.removed, mg.orb_score)]

        def score_partition(lab):
            return {"ari": float(adjusted_rand_score(g, lab)), "fmi": float(fowlkes_mallows_score(g, lab)), "n_groups": int(len(np.unique(lab)))}

        rows = []
        for thr in (None,) + THRESHOLDS:
            keep = [m for m in merges if thr is not None and m["orb_score"] >= thr]
            lab = join_pieces(pieces, names, saved_as, [(m["kept"], m["removed"]) for m in keep])
            row = {"threshold": thr, "n_pass": len(keep), "n_correct": sum(m["same_group"] for m in keep),
                   "n_wrong": sum(not m["same_group"] for m in keep), **{f"posthoc_{k}": v for k, v in score_partition(lab).items()}}
            if thr == 40:
                row["posthoc_equals_a1_final"] = bool(np.array_equal(lab, final_ref))
            rows.append(row)

        # extra: real replay of the ORB pass at each threshold (logged representative files, as approach12_score.py replay)
        rep_files = sorted(saved_as.values())
        rep_idx = [rep[f] for f in rep_files]
        feats = np.load(A12W / f"features_{cls}.npy")[rep_idx]
        paths = [A / "raw_resaved" / cls / f"{cls}_{i:05d}.jpeg" for i in rep_idx]
        orb = cv2.ORB_create(nfeatures=500)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        desc_cache, sharp_cache, score_cache = {}, {}, {}

        def desc(i):
            if i not in desc_cache:
                img = cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE)
                desc_cache[i] = None if img is None else orb.detectAndCompute(img, None)[1]
            return desc_cache[i]

        def score(i, j):
            if (i, j) not in score_cache:
                da, db = desc(i), desc(j)
                score_cache[(i, j)] = 0 if da is None or db is None else sum(1 for m in bf.match(da, db) if m.distance < 40)
            return score_cache[(i, j)]

        def sharp(i):
            if i not in sharp_cache:
                img = cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE)
                sharp_cache[i] = 0.0 if img is None else float(cv2.Laplacian(img, cv2.CV_64F).var())
            return sharp_cache[i]

        dists = np.linalg.norm(feats[:, None, :] - feats[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        k = min(10, len(rep_files) - 1)
        pairs = sorted({(min(i, j), max(i, j)) for i in range(len(rep_files)) for j in np.argsort(dists[i])[:k]})
        for row in rows:
            thr = row["threshold"]
            if thr is None:
                continue
            removed, done = set(), []
            for i, j in pairs:
                if i in removed or j in removed:
                    continue
                sc = score(i, j)
                if sc >= thr:
                    drop, kp = (j, i) if sharp(i) >= sharp(j) else (i, j)
                    removed.add(drop)
                    done.append((rep_files[kp], rep_files[drop], sc, bool(g[rep_idx[kp]] == g[rep_idx[drop]])))
            lab = join_pieces(pieces, names, saved_as, [(a, b) for a, b, _, _ in done])
            row.update({"replay_n_merges": len(done), "replay_n_correct": sum(d[3] for d in done), "replay_n_wrong": sum(not d[3] for d in done),
                        **{f"replay_{kk}": v for kk, v in score_partition(lab).items()}})
            if thr == 40:
                row["replay_equals_logged_merges"] = bool([(a, b, s_) for a, b, s_, _ in done] == [(m["kept"], m["removed"], m["orb_score"]) for m in merges])
        rows[0].update({"posthoc_" + k_: v for k_, v in score_partition(pieces).items()})
        out[cls] = {"merges": merges, "thresholds": rows}
        print(f"[orb] {cls}: {len(merges)} merges, same group {sum(m['same_group'] for m in merges)}, different {sum(not m['same_group'] for m in merges)}")
        for r_ in rows:
            print("   ", {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in r_.items()})
    (WORK / "orb.json").write_text(json.dumps(out, indent=1))


# --------------------------------------------------------------------------- #
# step report
# --------------------------------------------------------------------------- #
def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def references(cls: str) -> dict:
    m = load_mapping(cls)
    fa, names_a, _ = upstream_features(cls, "imagenet")
    fb, names_b, _ = upstream_features(cls, "resize_only")
    assert names_a == names_b
    row = {n: i for i, n in enumerate(names_a)}
    order = [row[f] for f in m["their_filename"]]
    assert len(set(order)) == N
    return {"lab": np.load(WORK / f"{cls}_lab_ref.npy"),
            "uni_imagenet_norm": np.load(ROOT / "embeddings" / ("" if cls == "colon_n" else cls) / "uni1_full.npy"),
            "uni_imagenet_raw": fa[order], "uni_resize_raw": fb[order], "uni_resize_norm": l2n(fb[order])}


def rotation_rows(cls: str):
    import pandas as pd
    from scipy.spatial.distance import cdist

    sel = json.loads((WORK / f"{cls}_selection.json").read_text())
    g = load_mapping(cls)["group_id"].to_numpy()
    sel_groups = g[sel["index"]]
    ref = references(cls)
    lab = np.load(WORK / f"{cls}_lab_conds.npz")
    uni = {cname(c): np.load(WORK / "uni" / f"{cls}_{cname(c)}.npz") for c in CONDS}
    spaces = [("lab", "primary", {cname(c): lab[cname(c)] for c in CONDS}, ref["lab"]),
              ("uni_imagenet", "primary", {k: l2n(v["imagenet"]) for k, v in uni.items()}, ref["uni_imagenet_norm"]),
              ("uni_resize_only", "primary", {k: l2n(v["resize_only"]) for k, v in uni.items()}, ref["uni_resize_norm"]),
              ("uni_imagenet", "raw_euclid_extra", {k: v["imagenet"] for k, v in uni.items()}, ref["uni_imagenet_raw"]),
              ("uni_resize_only", "raw_euclid_extra", {k: v["resize_only"] for k, v in uni.items()}, ref["uni_resize_raw"])]
    rows, checks = [], []
    for feature, space, feats, refm in spaces:
        q0 = feats["rot0"]
        d = cdist(q0, refm)
        other = g[None, :] != sel_groups[:, None]
        nearest = np.where(other, d, np.inf).min(axis=1)
        denom = float(np.median(nearest))
        diff = np.abs(q0 - refm[sel["index"]])
        checks.append({"section": "rotation_check", "cls": cls, "feature": feature, "space": space,
                       "max_abs_diff_rot0_vs_reference": float(diff.max()),
                       "max_rel_diff_rot0_vs_reference": float((np.linalg.norm(q0 - refm[sel["index"]], axis=1) / np.linalg.norm(refm[sel["index"]], axis=1)).max()),
                       "denominator_median_nearest_other": denom, "min_nearest_other": float(nearest.min())})
        for cond in CONDS:
            ds = np.linalg.norm(feats[cname(cond)] - q0, axis=1)
            ratio = ds / denom
            rows.append({"section": "rotation" if cond[0] != "zoom" else "rotation_control_zoom_only", "cls": cls, "feature": feature, "space": space,
                         "condition": cond[0], "angle": cond[1], "crop_px": crop_side(cond[1]) if cond[0] != "hflip" else SIDE, "n": N_SEL,
                         "median_ratio": float(np.median(ratio)), "mean_ratio": float(ratio.mean()), "p90_ratio": float(np.percentile(ratio, 90)),
                         "frac_exceeding_own_nearest_other": float((ds > nearest).mean()), "denominator_median_nearest_other": denom})
    return rows, checks, sel


def step_report() -> None:
    import pandas as pd

    OUTD.mkdir(parents=True, exist_ok=True)
    for f in ("testA.csv", "testB.csv", "NOTES.md"):
        if (OUTD / f).exists():
            die(f"{(OUTD / f).relative_to(ROOT)} exists, not overwriting")
    pk = json.loads((WORK / "pk.json").read_text())
    orb = json.loads((WORK / "orb.json").read_text())
    A_rows = []
    infos = {}
    for cls in CLASSES:
        rows, checks, sel = rotation_rows(cls)
        A_rows += rows + checks
        m = load_mapping(cls)
        for i, gr in zip(sel["index"], sel["groups"]):
            A_rows.append({"section": "selection", "cls": cls, "image_index": i, "my_filename": m["my_filename"][i], "published_group": gr})
        st = pk[cls]["stored"]
        for norm in ("imagenet", "resize_only"):
            for who in ("theirs", "mine"):
                for met, v in zip(("P@1", "P@5"), st[norm][who]):
                    A_rows.append({"section": "pk", "cls": cls, "feature": f"uni_{norm}_{who}_stored", "metric": met, "value": v})
            for met, key in (("P@1", "p1"), ("P@5", "p5")):
                A_rows.append({"section": "pk_validation", "cls": cls, "feature": f"uni_{norm}_copied_functions", "metric": met, "value": pk[cls]["validation"][norm][key],
                               "abs_diff_vs_stored_mine": pk[cls]["validation"][norm]["abs_diff_" + key]})
        for met, key in (("P@1", "p1"), ("P@5", "p5")):
            A_rows.append({"section": "pk", "cls": cls, "feature": "lab_histogram_this_work", "metric": met, "value": pk[cls]["lab"][key]})
        infos[cls] = json.loads((WORK / f"{cls}_lab_info.json").read_text())
    dfA = _ints(pd.DataFrame(A_rows))
    dfA.to_csv(OUTD / "testA.csv", index=False)

    B_rows = []
    for cls in CLASSES:
        ms = orb[cls]["merges"]
        for m_ in ms:
            B_rows.append({"section": "merge", "cls": cls, **m_})
        for kind, sub in (("same_group", [m_ for m_ in ms if m_["same_group"]]), ("different_group", [m_ for m_ in ms if not m_["same_group"]])):
            sc = np.array([m_["orb_score"] for m_ in sub]) if sub else np.array([np.nan])
            B_rows.append({"section": "score_stats", "cls": cls, "kind": kind, "n": len(sub), "score_min": float(np.nanmin(sc)), "score_median": float(np.nanmedian(sc)),
                           "score_max": float(np.nanmax(sc))})
        for r in orb[cls]["thresholds"]:
            B_rows.append({"section": "threshold", "cls": cls, **r})
    dfB = _ints(pd.DataFrame(B_rows))
    dfB.to_csv(OUTD / "testB.csv", index=False)
    (OUTD / "NOTES.md").write_text(build_notes(dfA, dfB, pk, infos))
    print("[report] wrote testA.csv, testB.csv, NOTES.md in", OUTD.relative_to(ROOT))


def reading(dfA, dfB, pk) -> list:
    """Short mechanical summary of the numbers above (every figure is computed from the tables)."""
    L = []
    w = L.append

    def get(c, sec, f, cond, ang, col="median_ratio", space="primary"):
        d = dfA[(dfA.cls == c) & (dfA.section == sec) & (dfA.feature == f) & (dfA.condition == cond) & (dfA.angle == ang) & (dfA.space == space)]
        return float(d.iloc[0][col])

    names = {"lab": "Lab", "uni_imagenet": "UNI imagenet", "uni_resize_only": "UNI resize-only"}
    w("## Reading of the numbers")
    w("")
    w("**Test A1.** Lab histograms re-identify their own group almost as well as UNI by P@1/P@5: "
      + "; ".join(f"{c} P@1 {pk[c]['lab']['p1']:.4f} / P@5 {pk[c]['lab']['p5']:.4f} (UNI resize-only, my reproduction: {pk[c]['stored']['resize_only']['mine'][0]:.4f} / {pk[c]['stored']['resize_only']['mine'][1]:.4f})" for c in CLASSES) + ".")
    w("")
    w("**Test A2.**")
    w("")
    for c in CLASSES:
        w(f"- {c}, 25 deg median ratio (rotation / zoom-only control): " + "; ".join(
            f"{names[f]} {get(c, 'rotation', f, 'rot', 25):.3f} / {get(c, 'rotation_control_zoom_only', f, 'zoom', 25):.3f}" for f in names)
          + f". Fraction of images moved past their nearest other original at 25 deg: " + ", ".join(f"{names[f]} {get(c, 'rotation', f, 'rot', 25, 'frac_exceeding_own_nearest_other'):.2f}" for f in names) + ".")
    fr = []
    for c in CLASSES:
        for f in names:
            for a in ANGLES[1:]:
                fr.append((get(c, "rotation_control_zoom_only", f, "zoom", a) / get(c, "rotation", f, "rot", a), c, f, a))
    lo, hi = min(fr), max(fr)
    w(f"- Zoom-only control: the crop and magnification alone give between {lo[0]:.0%} ({lo[1]}, {names[lo[2]]}, {lo[3]} deg) and {hi[0]:.0%} ({hi[1]}, {names[hi[2]]}, {hi[3]} deg) of the median ratio of the rotated images, over all classes, features and angles. "
      "So a large part of the displacement measured here comes from the crop and magnification (content removed, scale changed), not from the rotation; the rotation-only part is the difference between the two curves, and it is not separable beyond that with this design "
      "(a control rotating by multiples of 90 degrees, with no crop, would isolate it).")
    top = []
    for c in CLASSES:
        for f in names:
            first = [a for a in ANGLES[1:] if get(c, "rotation", f, "rot", a, "p90_ratio") >= 1.0]
            top.append(f"{c} {names[f]}: " + (f"from {first[0]} deg" if first else "never up to 25 deg"))
    w("- Median ratios stay below 1 for all three features at every angle up to 25 deg in both classes. First angle at which the 90th percentile of the ratio reaches 1: " + "; ".join(top) + ".")
    for c in CLASSES:
        w(f"- Horizontal flip, {c}: median ratio " + ", ".join(f"{names[f]} {get(c, 'rotation', f, 'hflip', 0):.3f}" for f in names) + " (a colour histogram does not change under a flip, UNI embeddings do).")
    w("")
    w("**Test B.**")
    w("")
    ss = dfB[dfB.section == "score_stats"]
    for c in CLASSES:
        s_ = ss[ss.cls == c].set_index("kind")
        th = dfB[(dfB.section == "threshold") & (dfB.cls == c)].set_index("threshold")
        w(f"- {c}: {int(s_.loc['same_group', 'n'])} merges join the same published group (ORB score {s_.loc['same_group', 'score_min']:g} to {s_.loc['same_group', 'score_max']:g}) and {int(s_.loc['different_group', 'n'])} join different groups "
          f"(score {s_.loc['different_group', 'score_min']:g} to {s_.loc['different_group', 'score_max']:g}). At threshold 60 no wrong merge passes (post-hoc {int(th.loc[60, 'n_correct'])} correct / {int(th.loc[60, 'n_wrong'])} wrong); "
          f"ARI {th.loc[40, 'posthoc_ari']:.4f} at the logged threshold 40, {th.loc[60, 'posthoc_ari']:.4f} post-hoc at 60 ({th.loc[60, 'replay_ari']:.4f} with a real replay), {th.iloc[0]['posthoc_ari']:.4f} with no merges.")
    w("")
    return L


def build_notes(dfA, dfB, pk, infos) -> str:
    L = []
    w = L.append
    w("# Discussion tests: rotation sensitivity (Test A) and ORB merges (Test B)")
    w("")
    w("Script: `scripts/discussion_tests.py` (steps lab, pk, uni, orb, report, plot). Lab and ORB work ran in `venv_sk_1.7.2` (opencv 5.1.0, sklearn 1.7.2), UNI in `venv` (timm 0.9.8, torch 2.14.0+cpu, CPU, fp32), the figure in `venv_repro`.")
    w("Data: the original files in `data/raw/<cls>`. Published groups from `embeddings/**/my_filename_to_group_id.csv`. Approach 1 logs and partitions from `data/approach12_drive/`.")
    w("")
    w("## Test A1: P@1 / P@5 in Lab-histogram space")
    w("")
    w("Upstream definition (`precision_at_1`, `precision_at_k`: Euclidean on the features as given, the image itself excluded, mean of the 0/1 same-published-group flag over all images and neighbours), copied verbatim into the script because the upstream module imports `umap`. The copy reproduces my stored UNI values from the stored features, per class: "
      + "; ".join(f"{c} {n}: P@1 {pk[c]['validation'][n]['p1']:.5f}, P@5 {pk[c]['validation'][n]['p5']:.5f} (stored {pk[c]['validation'][n]['stored_mine_p1']:.5f} / {pk[c]['validation'][n]['stored_mine_p5']:.5f})" for c in CLASSES for n in ("imagenet", "resize_only")) + ".")
    w("")
    w("| class | features | P@1 | P@5 |")
    w("|---|---|---|---|")
    for c in CLASSES:
        w(f"| {c} | **Lab histograms (96-d, approach 1 features; this work)** | {pk[c]['lab']['p1']:.5f} | {pk[c]['lab']['p5']:.5f} |")
        for n in ("resize_only", "imagenet"):
            for who, label in (("theirs", "published"), ("mine", "my reproduction")):
                a, b = pk[c]["stored"][n][who]
                w(f"| {c} | UNI {n} ({label}, stored) | {a:.5f} | {b:.5f} |")
    w("")
    w("## Test A2: controlled rotation")
    w("")
    sel = json.loads((WORK / "colon_n_selection.json").read_text())
    w(f"- Selection: `RandomState(0)` draws 100 distinct published groups per class, then one image per group (saved in `testA.csv`, section `selection`).")
    w(f"- Pipeline: original image, PIL `rotate(angle, BILINEAR, expand=False)` (black corners), centre-crop to the largest axis-aligned square without empty corners (side = 768/(cos+sin), floor, minus a 2 px margin, made even), bilinear resize back to 768. Crop sides: "
      + ", ".join(f"{a} deg {sel['crop_side'][str(a)]} px" for a in ANGLES) + ". The 0 deg condition runs through the same code (crop 768, no change).")
    w("- Features, all on the same in-memory image: Lab = approach 1 histogram (raw 96-d); UNI imagenet = the approach 3 / stage 2 timm transform (resize, centre crop, ImageNet normalisation); UNI resize-only = upstream `Resize(224)` on a uint8 tensor, scaled to [0,1], no normalisation.")
    w("- Ratio per image = distance from the transformed image's feature to the 0 deg feature of the same image, divided by one denominator per feature and class: the median over the 100 selected images of the distance from the unrotated image to its nearest image of a **different** published group (searched over all 5000 images of the class, same space). Primary space: Lab raw Euclidean; both UNI features L2-normalised Euclidean. Extra columns `raw_euclid_extra`: UNI without L2 normalisation (the reference for imagenet is then the upstream UNI/imagenet features, whose preprocessing differs slightly from the approach 3 transform).")
    w("- Also reported: the 90th percentile of the ratio, and the fraction of images whose transformed version is farther from its own 0 deg feature than that image's own nearest different-group image (i.e. the transform moved it past its nearest other original).")
    w("- **Control (zoom only).** The crop and resize back magnify the image (by 768/crop, up to 1.34x at 25 deg). `rotation_control_zoom_only` applies the same centre crop and resize to the unrotated image, to separate the effect of the zoom from the effect of the rotation.")
    w("- **Horizontal flip** at 0 deg (no crop) is its own row, `condition = hflip`.")
    w("")
    for c in CLASSES:
        d = dfA[(dfA.cls == c) & (dfA.section == "rotation") & (dfA.space == "primary")]
        w(f"### {c}: median ratio (p90 in brackets; fraction of images moved past their nearest other original)")
        w("")
        w("| condition | Lab | UNI imagenet | UNI resize-only |")
        w("|---|---|---|---|")
        order = [("rot", a) for a in ANGLES] + [("hflip", 0)]
        for cond in order:
            cells = []
            for f in ("lab", "uni_imagenet", "uni_resize_only"):
                r = d[(d.feature == f) & (d.condition == cond[0]) & (d.angle == cond[1])].iloc[0]
                cells.append(f"{r.median_ratio:.3f} ({r.p90_ratio:.3f}; {r.frac_exceeding_own_nearest_other:.2f})")
            w(f"| {'rotation ' + str(cond[1]) + ' deg' if cond[0] == 'rot' else 'horizontal flip'} | " + " | ".join(cells) + " |")
        w("")
        dz = dfA[(dfA.cls == c) & (dfA.section == "rotation_control_zoom_only") & (dfA.space == "primary")]
        w("Zoom-only control (same crop, no rotation), median ratio: " + "; ".join(
            f"{a} deg: " + " / ".join(f"{dz[(dz.feature == f) & (dz.angle == a)].iloc[0].median_ratio:.3f}" for f in ("lab", "uni_imagenet", "uni_resize_only")) for a in ANGLES[1:]) + " (Lab / UNI imagenet / UNI resize-only).")
        w("")
        ck = dfA[(dfA.cls == c) & (dfA.section == "rotation_check") & (dfA.space == "primary")]
        w("Denominators (median nearest-different-group distance): " + ", ".join(f"{r.feature} {r.denominator_median_nearest_other:.4g}" for r in ck.itertuples())
          + ". Checks, in-session 0 deg feature vs the stored reference row: max abs difference "
          + ", ".join(f"{r.feature} {r.max_abs_diff_rot0_vs_reference:.2e}" for r in ck.itertuples()) + ".")
        w("")
    w(f"Lab input check: PIL and cv2 decode of the originals give identical histograms (max abs difference "
      + ", ".join(f"{c} {infos[c]['pil_vs_cv2_decode_max_abs_diff_selected']:.1e}" for c in CLASSES)
      + "); the originals' histograms differ from approach 1's re-saved-image histograms by "
      + ", ".join(f"{c} mean {infos[c]['originals_vs_resaved_features_mean_abs_diff']:.1e} / max {infos[c]['originals_vs_resaved_features_max_abs_diff']:.1e}" for c in CLASSES) + ".")
    w("")
    w("Figure: `testA_rotation.png` (median ratio against angle, three features, horizontal-flip markers at 0 deg, reference line at 1).")
    w("")
    w("## Test B: ORB merges of approach 1")
    w("")
    w("Each logged merge (kept, removed, ORB score) is looked up by the published group of the two representative images (the logged representative file of each piece). Caveat: a merged piece can contain several published groups, so a merge that is correct at representative level can still join impure pieces.")
    w("")
    ss = dfB[dfB.section == "score_stats"]
    w("| class | kind | merges | ORB score min / median / max |")
    w("|---|---|---|---|")
    for r in ss.itertuples():
        w(f"| {r.cls} | {r.kind.replace('_', ' ')} | {int(r.n)} | {r.score_min:g} / {r.score_median:g} / {r.score_max:g} |")
    w("")
    w("Merges that would have passed each threshold (correct = same published group), and the ARI of the rebuilt a1 partition. `posthoc` applies only the logged merges at or above the threshold (union-find on the logged merges, as specified). **Extra, real replay:** the ORB pass re-run with that threshold (a merge removes a representative from later pairs, so the result differs from filtering after the fact); at threshold 40 both equal the logged result.")
    w("")
    w("| class | threshold | posthoc: correct / wrong | posthoc ARI | groups | extra, real replay: correct / wrong | replay ARI | groups |")
    w("|---|---|---|---|---|---|---|---|")
    th = dfB[dfB.section == "threshold"]
    for r in th.itertuples():
        thr = "no merges" if pd_isna(r.threshold) else f"{int(r.threshold)}"
        rep_cells = "n/a | n/a | n/a" if pd_isna(r.threshold) else f"{int(r.replay_n_correct)} / {int(r.replay_n_wrong)} | {r.replay_ari:.4f} | {int(r.replay_n_groups)}"
        w(f"| {r.cls} | {thr} | {int(r.n_correct)} / {int(r.n_wrong)} | {r.posthoc_ari:.4f} | {int(r.posthoc_n_groups)} | {rep_cells} |")
    w("")
    w("Checks: the post-hoc rebuild at threshold 40 equals `a1_final` exactly (" + ", ".join(f"{r.cls}: {bool(r.posthoc_equals_a1_final)}" for r in th[th.threshold == 40].itertuples())
      + "); the replay at threshold 40 equals the logged merge sequence (" + ", ".join(f"{r.cls}: {bool(r.replay_equals_logged_merges)}" for r in th[th.threshold == 40].itertuples()) + ").")
    w("")
    L += reading(dfA, dfB, pk)
    return "\n".join(L) + "\n"


def _ints(df):
    """Columns that hold only whole numbers (plus blanks) are written as integers, not 18.0."""
    for c in df.columns:
        col = df[c]
        if col.dtype == float and col.notna().any() and (col.dropna() % 1 == 0).all():
            df[c] = col.astype("Int64")
    return df


def pd_isna(x) -> bool:
    import pandas as pd

    return x is None or bool(pd.isna(x))


# --------------------------------------------------------------------------- #
# step plot (venv_repro: matplotlib)
# --------------------------------------------------------------------------- #
def step_plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D

    out = OUTD / "testA_rotation.png"
    if out.exists():
        die(f"{out.relative_to(ROOT)} exists, not overwriting")
    df = pd.read_csv(OUTD / "testA.csv")
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "font.size": 9, "axes.labelsize": 10,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
    feats = [("lab", "Lab histogram", "#2a78d6"), ("uni_imagenet", "UNI, ImageNet norm (approach 3)", "#eb6834"), ("uni_resize_only", "UNI, resize-only", "#1baf7a")]
    FLIP_X = {"lab": -0.9, "uni_imagenet": 0.0, "uni_resize_only": 0.9}  # horizontal offsets (degrees) so the three flip markers do not overlap
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, cls in zip(axes, CLASSES):
        d = df[(df.cls == cls) & (df.space == "primary")]
        for f, label, col in feats:
            r = d[(d.section == "rotation") & (d.feature == f) & (d.condition == "rot")].sort_values("angle")
            ax.plot(r.angle, r.median_ratio, "-o", color=col, lw=1.6, ms=5)
            h = d[(d.section == "rotation") & (d.feature == f) & (d.condition == "hflip")].iloc[0]
            ax.plot([FLIP_X[f]], [h.median_ratio], marker="D", color=col, ms=7, mec="white", mew=0.8, ls="", zorder=5)
        ax.axhline(1.0, color=INK2, lw=1.0, ls="--")
        ax.set_xlabel("Rotation angle (degrees)")
        ax.set_xticks(ANGLES)
        ax.text(0.02, 0.97, cls, transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK2)
        ax.set_xlim(-2, 26.5)
        ax.set_ylim(-0.03, 1.2)
    axes[0].set_ylabel("Distance to own unrotated image,\nin units of nearest other-group distance")
    axes[1].text(0.98, 1.015, "ratio 1 = as far as a different original", transform=axes[1].get_yaxis_transform(), ha="right", va="bottom", fontsize=7.5, color=INK2)
    h = [Line2D([], [], marker="o", ls="-", color=c, ms=5, label=lab) for _, lab, c in feats] + [Line2D([], [], marker="D", ls="", color=INK2, ms=6, label="horizontal flip (at 0 deg, offset sideways)")]
    axes[1].legend(handles=h, loc="lower right", fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print("[plot] wrote", out.relative_to(ROOT))


STEPS = {"lab": step_lab, "pk": step_pk, "uni": step_uni, "orb": step_orb, "report": step_report, "plot": step_plot}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
        die(f"usage: discussion_tests.py {{{','.join(STEPS)}}}")
    STEPS[sys.argv[1]]()
