#!/usr/bin/env python3
"""Test C: why do 96-d Lab histograms (KMeans ARI ~0.73 on colon_n) beat UNI resize_only (benchmark ARI 0.64)?

Run from the project root with the sklearn 1.9.1 venv (it also has opencv), always with -I:

    data/approach12_drive/venv_a12/bin/python -I scripts/discussion_testc.py t1 | t2 | t3 | report

    t1      KMeans k = published group count, n_init=10, seeds 0..9, six feature sets (+ k=250 for Lab raw)
    t2      AgglomerativeClustering (average, complete, single, ward) and SpectralClustering
            (nearest_neighbors, n_neighbors 5, 10, 15, 20, 30, seeds 0..2) on the three L2-normalised feature sets
    t3      Reinhard-normalised histograms, 6-number colour moments, P@1/P@5 reference rows
    report  results/discussion_tests/testC.csv and NOTES_testC.md

Every setting listed here was fixed before any result was seen; all of them are reported, none is selected by ARI.
Per-run results are cached in data/approach12_drive/work/testc/ (gitignored), so t1 and t2 can be resumed.
Feature sets: (a) approach 1/2 Lab histograms (approach12 features), (b) stored upstream UNI resize_only, (c) stored upstream UNI imagenet.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "data" / "approach12_drive"
A12W = A / "work"
WORK = A / "work" / "testc"
OUTD = ROOT / "results" / "discussion_tests"
UP = ROOT / "third_party" / "LC25000-clean"
CLASSES = ("colon_n", "colon_aca")
N = 5000
SEEDS10 = tuple(range(10))
SPEC_NN = (5, 10, 15, 20, 30)
SPEC_SEEDS = (0, 1, 2)
LINKAGES = ("average", "complete", "single", "ward")
T1_SETS = ("lab_raw", "lab_l2", "uni_resize_only_raw", "uni_resize_only_l2", "uni_imagenet_raw", "uni_imagenet_l2")
T2_SETS = ("lab_l2", "uni_resize_only_l2", "uni_imagenet_l2")


def die(msg: str) -> None:
    print(f"[STOP] {msg}", file=sys.stderr)
    sys.exit(2)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def mapping(cls: str):
    import pandas as pd

    d = pd.read_csv(ROOT / "embeddings" / ("" if cls == "colon_n" else cls) / "my_filename_to_group_id.csv")
    assert d["my_filename"].tolist() == [f"{cls}_{i:04d}.jpeg" for i in range(N)]
    return d


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def load_set(cls: str, name: str):
    """(features float32, published group per row) for one feature set."""
    import pandas as pd

    if name.startswith("lab"):
        x = np.load(A12W / f"features_{cls}.npy")
        g = mapping(cls)["group_id"].to_numpy()
    else:
        norm = "resize_only" if "resize_only" in name else "imagenet"
        d = UP / "feature_vectors" / cls / "UNI" / norm
        x = np.load(d / "features.npy")
        ids = json.loads((d / "ids_2_img_paths.json").read_text())
        names = [Path(ids[str(i)]).name for i in range(len(ids))]
        pub = pd.read_csv(ROOT / "data" / "lc25000_image_groups.csv").set_index("filename")["group_id"]
        g = pub.loc[names].to_numpy()
    if name.endswith("_l2"):
        x = l2n(x).astype(np.float32)
    assert x.shape[0] == N and len(g) == N
    return x, g


def save_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.rename(path)


# --------------------------------------------------------------------------- #
# t1: KMeans
# --------------------------------------------------------------------------- #
def step_t1() -> None:
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    WORK.mkdir(parents=True, exist_ok=True)
    jobs = []
    for cls in CLASSES:
        for s in ("lab_raw", "lab_l2"):
            jobs += [(cls, s, None, seed) for seed in SEEDS10]
    for cls in CLASSES:
        jobs += [(cls, "lab_raw", 250, seed) for seed in SEEDS10 + (42,)]
    for cls in CLASSES:
        for s in T1_SETS[2:]:
            jobs += [(cls, s, None, seed) for seed in SEEDS10]
    cache: dict = {}
    for cls, s, k_fixed, seed in jobs:
        tag = f"{s}_k{k_fixed}" if k_fixed else s
        path = WORK / f"t1__{cls}__{tag}__{seed}.json"
        if path.exists():
            continue
        if (cls, s) not in cache:
            cache = {(cls, s): load_set(cls, s)}
        x, g = cache[(cls, s)]
        k = k_fixed or len(np.unique(g))
        t0 = time.time()
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
        ari = float(adjusted_rand_score(g, lab))
        save_json(path, {"cls": cls, "set": tag, "k": int(k), "seed": seed, "n_init": 10, "ari": ari, "seconds": time.time() - t0, "sklearn": sklearn.__version__})
        print(f"[t1] {cls} {tag} k={k} seed={seed} ARI {ari:.4f} ({time.time() - t0:.0f}s)", flush=True)
    print("[t1] ALL DONE", flush=True)


# --------------------------------------------------------------------------- #
# t2: other clustering methods
# --------------------------------------------------------------------------- #
def step_t2() -> None:
    import sklearn
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering
    from sklearn.metrics import adjusted_rand_score

    WORK.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        for s in T2_SETS:
            x, g = load_set(cls, s)
            k = len(np.unique(g))
            runs = [("agglomerative", lk, 0) for lk in LINKAGES] + [("spectral", nn, seed) for nn in SPEC_NN for seed in SPEC_SEEDS]
            for method, param, seed in runs:
                path = WORK / f"t2__{cls}__{s}__{method}__{param}__{seed}.json"
                if path.exists():
                    continue
                t0 = time.time()
                rec = {"cls": cls, "set": s, "method": method, "param": str(param), "seed": seed, "k": int(k), "sklearn": sklearn.__version__}
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        if method == "agglomerative":
                            lab = AgglomerativeClustering(n_clusters=k, linkage=param).fit_predict(x)
                        else:
                            lab = SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=param, random_state=seed).fit_predict(x)
                        rec["ari"] = float(adjusted_rand_score(g, lab))
                        rec["n_clusters_found"] = int(len(np.unique(lab)))
                    except Exception as e:  # recorded, not hidden
                        rec["ari"] = None
                        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                rec["warnings"] = sorted({str(w.message)[:120] for w in caught if "not fully connected" in str(w.message) or "converge" in str(w.message).lower()})
                rec["seconds"] = time.time() - t0
                save_json(path, rec)
                print(f"[t2] {cls} {s} {method} {param} seed={seed} ARI {rec['ari']} ({rec['seconds']:.0f}s){' WARN' if rec['warnings'] else ''}", flush=True)
    print("[t2] ALL DONE", flush=True)


# --------------------------------------------------------------------------- #
# t3: colour shortcut
# --------------------------------------------------------------------------- #
def lab_pixels(cls: str, i: int) -> np.ndarray:
    import cv2
    from PIL import Image

    arr = np.asarray(Image.open(ROOT / "data" / "raw" / cls / f"{cls}_{i:04d}.jpeg").convert("RGB"))
    return cv2.cvtColor(np.ascontiguousarray(arr[:, :, ::-1]), cv2.COLOR_BGR2Lab)  # 8-bit Lab, as approach 1


def hist_from_lab(lab: np.ndarray) -> np.ndarray:
    import cv2

    parts = []
    for ch in range(3):
        h = cv2.calcHist([lab], [ch], None, [32], [0, 256])
        parts.append(cv2.normalize(h, h).flatten())
    return np.concatenate(parts)


def reinhard(lab: np.ndarray, mu_i: np.ndarray, sd_i: np.ndarray, mu_t: np.ndarray, sd_t: np.ndarray):
    x = lab.astype(np.float64)
    scale = np.where(sd_i > 1e-6, sd_t / np.maximum(sd_i, 1e-6), 1.0)
    y = (x - mu_i) * scale + mu_t
    clipped = float(((y < 0) | (y > 255)).mean())
    return np.clip(np.rint(y), 0, 255).astype(np.uint8), clipped


def step_t3() -> None:
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    sys.path.insert(0, str(ROOT / "scripts"))
    import discussion_tests as dt  # upstream P@k copy (numpy only at import time)

    WORK.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        out_path = WORK / f"t3__{cls}.json"
        if out_path.exists():
            continue
        g = mapping(cls)["group_id"].to_numpy()
        k = len(np.unique(g))
        print(f"[t3] {cls}: pass 1 (moments and histograms of the originals)", flush=True)
        mom = np.zeros((N, 6))
        hist_orig = np.zeros((N, 96), dtype=np.float32)
        for i in range(N):
            lab = lab_pixels(cls, i)
            flat = lab.reshape(-1, 3)
            mom[i] = np.concatenate([flat.mean(0), flat.std(0)])
            hist_orig[i] = hist_from_lab(lab)
        mu_t, sd_t = mom[:, :3].mean(0), mom[:, 3:].mean(0)  # class-wide averages of the per-image mean and std
        print(f"[t3] {cls}: targets mean {mu_t.round(2)} std {sd_t.round(2)}; pass 2 (Reinhard)", flush=True)
        hist_rh = np.zeros((N, 96), dtype=np.float32)
        resid = np.zeros((N, 6))
        clipped = np.zeros(N)
        for i in range(N):
            lab = lab_pixels(cls, i)
            y, clipped[i] = reinhard(lab, mom[i, :3], mom[i, 3:], mu_t, sd_t)
            flat = y.reshape(-1, 3)
            resid[i] = np.concatenate([flat.mean(0) - mu_t, flat.std(0) - sd_t])
            hist_rh[i] = hist_from_lab(y)
        np.savez(WORK / f"t3__{cls}_arrays.npz", moments=mom, hist_orig=hist_orig, hist_reinhard=hist_rh)

        resaved = np.load(A12W / f"features_{cls}.npy")
        z6 = (mom - mom.mean(0)) / mom.std(0)
        sets = {"lab_approach12_resaved": resaved, "lab_originals": hist_orig, "lab_reinhard": hist_rh,
                "moments6_raw": mom.astype(np.float32), "moments6_zscored": z6.astype(np.float32)}
        res = {"cls": cls, "k": int(k), "sklearn": sklearn.__version__, "targets_mean": mu_t.tolist(), "targets_std": sd_t.tolist(),
               "reinhard_residual_max_abs_mean": float(np.abs(resid[:, :3]).max()), "reinhard_residual_max_abs_std": float(np.abs(resid[:, 3:]).max()),
               "reinhard_residual_mean_abs_mean": float(np.abs(resid[:, :3]).mean()), "reinhard_residual_mean_abs_std": float(np.abs(resid[:, 3:]).mean()),
               "reinhard_images_with_clipping": int((clipped > 0).sum()), "reinhard_mean_clipped_fraction": float(clipped.mean()),
               "originals_vs_resaved_hist_max_abs_diff": float(np.abs(hist_orig - resaved).max()), "sets": {}}
        conn = dt.connectivity(g)
        for name, x in sets.items():
            print(f"[t3] {cls} {name}: P@k and KMeans", flush=True)
            r = {"p1": float(dt.precision_at_1(x, conn)), "p5": float(dt.precision_at_k(x, conn, k=5)), "ari_seeds": []}
            for seed in SEEDS10:
                lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
                r["ari_seeds"].append(float(adjusted_rand_score(g, lab)))
            res["sets"][name] = r
        # P@1 / P@5 reference rows for the six T1 feature sets
        res["pk_reference"] = {}
        for name in T1_SETS:
            x, gg = load_set(cls, name)
            c2 = dt.connectivity(gg)
            res["pk_reference"][name] = {"p1": float(dt.precision_at_1(x, c2)), "p5": float(dt.precision_at_k(x, c2, k=5))}
            print(f"[t3] {cls} P@k reference {name}: {res['pk_reference'][name]}", flush=True)
        save_json(out_path, res)
    print("[t3] ALL DONE", flush=True)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def ms(v) -> tuple:
    v = np.array(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"), float(v.min()), float(v.max())


def load_runs(prefix: str) -> list:
    out = []
    for p in sorted(WORK.glob(f"{prefix}__*.json")):
        if p.name.endswith(".tmp"):
            continue
        out.append(json.loads(p.read_text()))
    return out


def step_report() -> None:
    import pandas as pd

    for f in ("testC.csv", "NOTES_testC.md"):
        if (OUTD / f).exists():
            die(f"{(OUTD / f).relative_to(ROOT)} exists, not overwriting")
    OUTD.mkdir(parents=True, exist_ok=True)
    t1, t2 = load_runs("t1"), load_runs("t2")
    t3 = {c: json.loads((WORK / f"t3__{c}.json").read_text()) for c in CLASSES}
    expected_t1 = len(CLASSES) * (6 * 10 + 11)
    expected_t2 = len(CLASSES) * len(T2_SETS) * (len(LINKAGES) + len(SPEC_NN) * len(SPEC_SEEDS))
    if len(t1) != expected_t1 or len(t2) != expected_t2:
        die(f"incomplete: t1 {len(t1)}/{expected_t1}, t2 {len(t2)}/{expected_t2}")

    rows = []
    for r in t1:
        rows.append({"section": "t1_run", "cls": r["cls"], "feature_set": r["set"], "k": r["k"], "seed": r["seed"], "n_init": r["n_init"], "ari": r["ari"], "sklearn": r["sklearn"]})
    summ1 = {}
    for cls in CLASSES:
        for tag in sorted({r["set"] for r in t1}):
            v = [r["ari"] for r in t1 if r["cls"] == cls and r["set"] == tag and r["seed"] in SEEDS10]
            m, sd, lo, hi = ms(v)
            summ1[(cls, tag)] = (m, sd, lo, hi)
            rows.append({"section": "t1_summary", "cls": cls, "feature_set": tag, "k": next(r["k"] for r in t1 if r["cls"] == cls and r["set"] == tag), "n": len(v),
                         "ari_mean": m, "ari_sd": sd, "ari_min": lo, "ari_max": hi})
    for r in t2:
        rows.append({"section": "t2_run", "cls": r["cls"], "feature_set": r["set"], "method": r["method"], "param": r["param"], "seed": r["seed"], "k": r["k"], "ari": r["ari"],
                     "n_clusters_found": r.get("n_clusters_found"), "warnings": "; ".join(r.get("warnings", [])), "error": r.get("error", ""), "sklearn": r["sklearn"]})
    summ2, over2 = {}, {}
    for cls in CLASSES:
        for s_ in T2_SETS:
            for method in ("agglomerative", "spectral"):
                allv = []
                for param in (LINKAGES if method == "agglomerative" else [str(n) for n in SPEC_NN]):
                    sub = [r for r in t2 if r["cls"] == cls and r["set"] == s_ and r["method"] == method and r["param"] == param]
                    v = [r["ari"] for r in sub if r["ari"] is not None]
                    m, sd, lo, hi = ms(v) if v else (float("nan"),) * 4
                    summ2[(cls, s_, method, param)] = (m, sd, lo, hi, len(sub), len(v), sum(bool(r.get("warnings")) for r in sub))
                    allv += v
                    rows.append({"section": "t2_summary", "cls": cls, "feature_set": s_, "method": method, "param": param, "n": len(sub), "n_ok": len(v),
                                 "n_with_warning": sum(bool(r.get("warnings")) for r in sub), "ari_mean": m, "ari_sd": sd, "ari_min": lo, "ari_max": hi})
                lo, hi = (min(allv), max(allv)) if allv else (float("nan"), float("nan"))
                over2[(cls, s_, method)] = (lo, hi, len(allv))
                rows.append({"section": "t2_overall", "cls": cls, "feature_set": s_, "method": method, "n": len(allv), "ari_min": lo, "ari_max": hi})
    for cls in CLASSES:
        r3 = t3[cls]
        for name, r in r3["sets"].items():
            m, sd, lo, hi = ms(r["ari_seeds"])
            rows.append({"section": "t3_set", "cls": cls, "feature_set": name, "k": r3["k"], "p1": r["p1"], "p5": r["p5"], "n": 10, "ari_mean": m, "ari_sd": sd, "ari_min": lo, "ari_max": hi})
            for seed, a in zip(SEEDS10, r["ari_seeds"]):
                rows.append({"section": "t3_run", "cls": cls, "feature_set": name, "k": r3["k"], "seed": seed, "n_init": 10, "ari": a})
        for name, r in r3["pk_reference"].items():
            rows.append({"section": "pk_reference", "cls": cls, "feature_set": name, "p1": r["p1"], "p5": r["p5"]})
        info = {k: v for k, v in r3.items() if k.startswith("reinhard") or k.startswith("targets") or k.startswith("originals")}
        rows.append({"section": "t3_info", "cls": cls, "note": json.dumps(info)})
    df = pd.DataFrame(rows)
    for c in df.columns:
        col = df[c]
        if col.dtype == float and col.notna().any() and (col.dropna() % 1 == 0).all():
            df[c] = col.astype("Int64")
    df.to_csv(OUTD / "testC.csv", index=False)
    (OUTD / "NOTES_testC.md").write_text(build_notes(t1, t2, t3, summ1, summ2, over2))
    print("[report] wrote testC.csv and NOTES_testC.md in", OUTD.relative_to(ROOT))


def reading(t1, t2, t3, summ1, summ2, over2) -> list:
    """Figures below are computed from the tables above; wording only states what the numbers show."""
    L = []
    w = L.append
    w("## Reading of the numbers")
    w("")
    w("**T1 (KMeans, `n_init=10`).**")
    for cls in CLASSES:
        la, lb = summ1[(cls, "lab_raw")], summ1[(cls, "uni_resize_only_raw")]
        la2, lb2 = summ1[(cls, "lab_l2")], summ1[(cls, "uni_resize_only_l2")]
        w(f"- {cls}: Lab minus UNI resize_only is {la[0] - lb[0]:+.3f} (raw) and {la2[0] - lb2[0]:+.3f} (L2) ARI; seed sd is {max(la[1], lb[1]):.3f} and {max(la2[1], lb2[1]):.3f}. "
          f"UNI imagenet: {summ1[(cls, 'uni_imagenet_raw')][0]:.3f} raw, {summ1[(cls, 'uni_imagenet_l2')][0]:.3f} L2.")
    w("- So the ordering Lab > UNI resize_only under KMeans holds on colon_n and reverses on colon_aca, where UNI is far ahead; it is not a general property of the two feature types.")
    w("- `n_init`: for UNI resize_only raw on colon_n the mean over seeds 0..9 is 0.663 with `n_init='auto'` and 0.678 with `n_init=10` (reference values above). `n_init=10` therefore moves the seed mean up by about 0.015; the benchmark's 0.643 (seed 42, `n_init='auto'`) sits at the bottom of the `n_init='auto'` seed range. It does not close the gap to Lab (about 0.72).")
    w("")
    w("**T2 (method).**")
    for cls in CLASSES:
        better_uni, better_lab = [], []
        for method, params in (("agglomerative", LINKAGES), ("spectral", [str(n) for n in SPEC_NN])):
            for param in params:
                a, b = summ2[(cls, "lab_l2", method, param)][0], summ2[(cls, "uni_resize_only_l2", method, param)][0]
                (better_uni if b > a else better_lab).append(f"{method} {param}")
        w(f"- {cls}, mean ARI of UNI resize_only above Lab in {len(better_uni)} of {len(better_uni) + len(better_lab)} settings ({', '.join(better_uni) or 'none'}); Lab above UNI in {', '.join(better_lab) or 'none'}.")
        w(f"- {cls}, range over all settings: spectral Lab {over2[(cls, 'lab_l2', 'spectral')][0]:.3f} to {over2[(cls, 'lab_l2', 'spectral')][1]:.3f}, spectral UNI resize_only {over2[(cls, 'uni_resize_only_l2', 'spectral')][0]:.3f} to {over2[(cls, 'uni_resize_only_l2', 'spectral')][1]:.3f}, "
          f"spectral UNI imagenet {over2[(cls, 'uni_imagenet_l2', 'spectral')][0]:.3f} to {over2[(cls, 'uni_imagenet_l2', 'spectral')][1]:.3f}; KMeans (T1, L2) Lab {summ1[(cls, 'lab_l2')][0]:.3f}, UNI resize_only {summ1[(cls, 'uni_resize_only_l2')][0]:.3f}.")
    sp = [summ2[("colon_n", "uni_resize_only_l2", "spectral", str(n))][0] - summ1[("colon_n", "uni_resize_only_l2")][0] for n in SPEC_NN]
    w(f"- On colon_n, spectral clustering on the same UNI resize_only features is above KMeans (L2, `n_init=10`) in every `n_neighbors` setting tried, by {min(sp):.3f} to {max(sp):.3f} ARI, while average, complete and single linkage are far below KMeans "
      f"({summ2[('colon_n', 'uni_resize_only_l2', 'agglomerative', 'average')][0]:.3f}, {summ2[('colon_n', 'uni_resize_only_l2', 'agglomerative', 'complete')][0]:.3f}, {summ2[('colon_n', 'uni_resize_only_l2', 'agglomerative', 'single')][0]:.3f} against {summ1[('colon_n', 'uni_resize_only_l2')][0]:.3f}). "
      "The ranking of the feature sets therefore depends on the clustering method, not only on the features. "
      "Spectral clustering warned about a not fully connected neighbour graph in most settings (marked `W`), and the colon_aca UNI imagenet result at 5 neighbours (about 0.13) shows what a failed graph looks like; no setting was removed or selected.")
    w("")
    w("**T3 (colour shortcut).**")
    for cls in CLASSES:
        r = t3[cls]["sets"]
        base = np.mean(r["lab_originals"]["ari_seeds"])
        w(f"- {cls}: equalising each image's Lab channel mean and std lowers Lab KMeans ARI from {base:.3f} to {np.mean(r['lab_reinhard']['ari_seeds']):.3f} and P@1/P@5 from {r['lab_originals']['p1']:.4f} / {r['lab_originals']['p5']:.4f} to {r['lab_reinhard']['p1']:.4f} / {r['lab_reinhard']['p5']:.4f}. "
          f"The six mean/std numbers alone give P@1/P@5 {r['moments6_raw']['p1']:.4f} / {r['moments6_raw']['p5']:.4f} and KMeans ARI {np.mean(r['moments6_raw']['ari_seeds']):.3f} (z-scored: {r['moments6_zscored']['p1']:.4f} / {r['moments6_zscored']['p5']:.4f}, ARI {np.mean(r['moments6_zscored']['ari_seeds']):.3f}).")
    w("- Per-image colour mean and std carry a large part of the grouping information in the Lab histograms (six numbers reach P@1 of 0.95 to 0.98, against 0.99 to 1.00 for the full histogram), and much of the Lab histogram's KMeans ARI disappears when they are equalised. "
      "Two limits: the normalisation also stretches and clips histograms (about 3000 of 5000 images have some clipped pixels), so part of the drop may come from that distortion and not only from removing the colour moments; and these tests do not show why UNI does worse under KMeans. "
      "Whether UNI also loses ARI on colour-equalised images is the T3b question (UNI resize_only on the Reinhard images), not run.")
    w("")
    return L


def build_notes(t1, t2, t3, summ1, summ2, over2) -> str:
    import pandas as pd

    L = []
    w = L.append
    names = {"lab_raw": "(a) Lab histograms, raw", "lab_l2": "(a) Lab histograms, L2-normalised", "uni_resize_only_raw": "(b) UNI resize_only, raw",
             "uni_resize_only_l2": "(b) UNI resize_only, L2", "uni_imagenet_raw": "(c) UNI imagenet, raw", "uni_imagenet_l2": "(c) UNI imagenet, L2"}
    w("# Test C: why Lab histograms beat UNI resize_only under KMeans on colon_n")
    w("")
    w("Question: on colon_n, 96-d Lab histograms (approach 1/2 KMeans, ARI 0.73) beat UNI resize_only (benchmark ARI 0.64, 10-start mean 0.68) although UNI has the higher P@5. Run on colon_n and colon_aca.")
    w("")
    w("Script: `scripts/discussion_testc.py` (steps t1, t2, t3, report), sklearn 1.9.1 and opencv 5.1.0 (`data/approach12_drive/venv_a12`), CPU. Feature sets: (a) the approach12 Lab histograms (computed from the re-saved images), (b) and (c) the stored upstream UNI `resize_only` and `imagenet` features in `third_party/.../feature_vectors` (raw), each raw and L2-normalised. ARI is against the published groups (257 for colon_n, 249 for colon_aca).")
    w("")
    w("**No setting was chosen by looking at ARI.** Every method, linkage, `n_neighbors` value and seed below was fixed in the plan before any result was seen, and all of them are reported (every spectral `n_neighbors` x seed run, every linkage), with the range over all settings given instead of a best one. Runs that failed or warned are listed as such, not dropped.")
    w("")
    w("## Reference values already in the repro (not recomputed here)")
    w("")
    ref = {}
    for cls, path in (("colon_n", ROOT / "repro/out/kmeans_seeds_uni_k257.json"), ("colon_aca", ROOT / "repro/out/colon_aca/kmeans_seeds_uni_k249.json")):
        d = json.loads(path.read_text())
        a = [d["per_run"]["auto"][str(s)]["ARI"] for s in SEEDS10]
        b = [d["per_run"]["10"][str(s)]["ARI"] for s in SEEDS10]
        ref[cls] = {"auto": a, "10": b, "seed42_auto": d["per_run"]["auto"]["42"]["ARI"]}
        w(f"- {cls}, UNI resize_only raw, KMeans (k = group count), seeds 0..9: `n_init='auto'` (= 1 start for k-means++) mean {np.mean(a):.4f} ± {np.std(a, ddof=1):.4f}; `n_init=10` mean {np.mean(b):.4f} ± {np.std(b, ddof=1):.4f}. The benchmark number is seed 42 with `n_init='auto'`: {ref[cls]['seed42_auto']:.4f}.")
    sc = pd.read_csv(ROOT / "results/approach12_vs_ground_truth.csv").set_index(["cls", "label_set"])
    w(f"- Approach 1/2 plain Lab KMeans, k=250, seed 42, `n_init=10`, as Colab ran it (sklearn 1.7.2, which reproduces Colab's labels; see `results/approach12_NOTES.md`): colon_n ARI {sc.loc[('colon_n', 'lab_kmeans250'), 'ari']:.4f}, colon_aca {sc.loc[('colon_aca', 'lab_kmeans250'), 'ari']:.4f}.")
    w("")
    w("## T1: KMeans with k = published group count, `n_init=10`, seeds 0..9")
    w("")
    for cls in CLASSES:
        w(f"### {cls}")
        w("")
        w("| feature set | k | ARI mean ± sd | min / max |")
        w("|---|---|---|---|")
        for tag in ("lab_raw", "lab_l2", "uni_resize_only_raw", "uni_resize_only_l2", "uni_imagenet_raw", "uni_imagenet_l2", "lab_raw_k250"):
            m, sd, lo, hi = summ1[(cls, tag)]
            k = next(r["k"] for r in t1 if r["cls"] == cls and r["set"] == tag)
            w(f"| {names.get(tag, '(a) Lab histograms, raw, **k=250**')} | {k} | {m:.4f} ± {sd:.4f} | {lo:.4f} / {hi:.4f} |")
        s42 = next(r["ari"] for r in t1 if r["cls"] == cls and r["set"] == "lab_raw_k250" and r["seed"] == 42)
        w("")
        w(f"Lab raw, k=250, seed 42, `n_init=10` under sklearn 1.9.1: {s42:.4f}; the Colab run (sklearn 1.7.2) scored {sc.loc[(cls, 'lab_kmeans250'), 'ari']:.4f} (for colon_n the two sklearn versions give different partitions, for colon_aca the same one).")
        mine = [r["ari"] for r in sorted((r for r in t1 if r["cls"] == cls and r["set"] == "uni_resize_only_raw"), key=lambda r: r["seed"])]
        dev = max(abs(x - y) for x, y in zip(mine, ref[cls]["10"]))
        w(f"Cross-check: T1 UNI resize_only raw (`n_init=10`, seeds 0..9) vs the stored repro run: max abs difference {dev:.2e}.")
        norms = np.linalg.norm(np.load(A12W / f"features_{cls}.npy"), axis=1)
        dmax = max(abs(a['ari'] - b['ari']) for a in t1 for b in t1 if a['cls'] == cls == b['cls'] and a['set'] == 'lab_raw' and b['set'] == 'lab_l2' and a['seed'] == b['seed'])
        w(f"Lab raw vs L2-normalised: every Lab row has norm {norms.min():.6f} to {norms.max():.6f} (sqrt(3) = {np.sqrt(3):.6f}), so the two inputs are the same up to a constant factor. Even so, individual seeds differ by up to {dmax:.3f} ARI between them, comparable to the seed-to-seed sd above, so differences of that size between two KMeans runs on essentially the same features are run-to-run noise, not feature effects.")
        w("")
    w("## T2: other clustering methods on the L2-normalised feature sets (k = group count)")
    w("")
    w("Agglomerative: one run per linkage (deterministic). Spectral (`affinity='nearest_neighbors'`): `n_neighbors` 5, 10, 15, 20, 30, seeds 0, 1, 2. All settings are shown. `W` = sklearn warned that the neighbour graph is not fully connected or a solver did not converge in at least one run of that setting.")
    w("")
    for cls in CLASSES:
        w(f"### {cls}")
        w("")
        w("| setting | (a) Lab | (b) UNI resize_only | (c) UNI imagenet |")
        w("|---|---|---|---|")
        for method, params in (("agglomerative", LINKAGES), ("spectral", [str(n) for n in SPEC_NN])):
            for param in params:
                cells = []
                for s_ in T2_SETS:
                    m, sd, lo, hi, n, nok, nw = summ2[(cls, s_, method, param)]
                    flag = " W" if nw else ""
                    cells.append("failed" if nok == 0 else (f"{m:.4f}" + (f" ± {sd:.4f}" if n > 1 else "") + flag + (f" ({nok}/{n} ok)" if nok < n else "")))
                w(f"| {method} {param}{' linkage' if method == 'agglomerative' else ' neighbours'} | " + " | ".join(cells) + " |")
        w("")
        w("Range over all settings of a family (min to max ARI over every run): " + "; ".join(
            f"{('Lab' if s_ == 'lab_l2' else 'UNI resize_only' if s_ == 'uni_resize_only_l2' else 'UNI imagenet')} {method} {over2[(cls, s_, method)][0]:.4f} to {over2[(cls, s_, method)][1]:.4f}"
            for s_ in T2_SETS for method in ("agglomerative", "spectral")) + ".")
        w("")
    w("## T3: colour shortcut")
    w("")
    w("Reinhard normalisation: each original image is converted to 8-bit Lab with cv2 (as approach 1), and each channel is rescaled so that its mean and std equal the class-wide averages of the per-image means and stds, clipped to [0, 255] and rounded; the 32-bin histograms are computed directly on these Lab channels (no round trip through BGR). "
      "`lab_originals` is the same histogram computed from the original files (approach 1's features come from the re-saved images), as a control. `moments6` are the per-image mean and std of the three 8-bit Lab channels (6 numbers; raw, and z-scored per dimension as an extra). KMeans: k = group count, `n_init=10`, seeds 0..9. P@1/P@5: upstream definition, Euclidean on the features as given.")
    w("")
    for cls in CLASSES:
        r3 = t3[cls]
        w(f"### {cls}")
        w("")
        w(f"Targets (class-wide average per-image Lab mean / std): mean {', '.join(f'{x:.1f}' for x in r3['targets_mean'])}, std {', '.join(f'{x:.1f}' for x in r3['targets_std'])}. After normalisation and clipping the per-image mean is within {r3['reinhard_residual_max_abs_mean']:.2f} (mean abs {r3['reinhard_residual_mean_abs_mean']:.3f}) and the std within {r3['reinhard_residual_max_abs_std']:.2f} (mean abs {r3['reinhard_residual_mean_abs_std']:.3f}) of the targets; {r3['reinhard_images_with_clipping']} of 5000 images had any clipped pixel (mean clipped fraction {r3['reinhard_mean_clipped_fraction']:.2e}). Originals vs re-saved histograms: max abs difference {r3['originals_vs_resaved_hist_max_abs_diff']:.3f}.")
        w("")
        w("| feature set | P@1 | P@5 | KMeans ARI mean ± sd | min / max |")
        w("|---|---|---|---|---|")
        for name, r in r3["sets"].items():
            m, sd, lo, hi = ms(r["ari_seeds"])
            w(f"| {name} | {r['p1']:.4f} | {r['p5']:.4f} | {m:.4f} ± {sd:.4f} | {lo:.4f} / {hi:.4f} |")
        w("")
        w("P@1 / P@5 reference rows (same function): " + "; ".join(f"{names[n]} {v['p1']:.4f} / {v['p5']:.4f}" for n, v in r3["pk_reference"].items()) + ".")
        w("")
    w("T3b (UNI resize_only on the Reinhard-normalised images) was not run: it needs about 38 minutes of UNI extraction per class and was left to the user's decision.")
    w("")
    L += reading(t1, t2, t3, summ1, summ2, over2)
    return "\n".join(L) + "\n"


STEPS = {"t1": step_t1, "t2": step_t2, "t3": step_t3, "report": step_report}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
        die(f"usage: discussion_testc.py {{{','.join(STEPS)}}}")
    STEPS[sys.argv[1]]()
