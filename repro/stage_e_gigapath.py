"""Prov-GigaPath features from Colab: place them in the upstream tree, then score them like the other 11 combos.

    venv_repro/bin/python repro/stage_e_gigapath.py place --cls colon_aca
    # evaluation (from an empty scratch dir, so upstream eval_results is never written): see repro/REPORT_colon_aca.md
    venv_repro/bin/python repro/stage_e_gigapath.py row --cls colon_aca

place  verifies data/gigapath_out/<cls>/ against repro/gigapath_colab_sha256.txt, checks the features and that
       the id -> path mapping equals the UNI resize_only one, and copies (never overwrites) the three files to
       upstream feature_vectors/<cls>/prov-gigapath/imagenet/.
row    compares the scratch evaluation row with their published prov-gigapath row (same bars as
       stage_d_compare_benchmark.py) and adds the 10-seed KMeans spread (as stage_d_kmeans_seeds_all.py).
       Writes new files only, in repro/out/<cls>/ (the script stops if one exists).
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cls import CLASSES, ROOT, UP, Cls  # noqa: E402

SRC = f'{ROOT}/data/gigapath_out'
SHA_FILE = f'{ROOT}/repro/gigapath_colab_sha256.txt'
FILES = ['features.npy', 'ids.npy', 'ids_2_img_paths.json']
KEY = 'prov-gigapath#imagenet#euclidean#NoReduction#kmeans'
COMBO = 'prov-gigapath#imagenet'
COLS = {'Adjusted Rand Index (ARI)': 'ARI', 'Fowlkes-Mallows Index': 'FMI', 'Normalized Mutual Information (NMI)': 'NMI',
        'Homogeneity': 'Homog', 'Completeness': 'Compl', 'precision@1': 'P@1', 'precision@5': 'P@5'}
BARS = {'ARI': 0.005, 'FMI': 0.005, 'P@1': 0.002, 'P@5': 0.002}   # CRITERIA.md C2


def die(msg):
    sys.exit(f'ERROR: {msg}')


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def expected():
    out, provenance = {}, []
    for line in open(SHA_FILE).read().splitlines():
        if line.startswith('#'):
            provenance.append(line.lstrip('# ').strip())
        elif line.strip():
            digest, rel = line.split()
            out[rel] = digest
    return out, provenance


def dest_dir(cls):
    return f'{UP}/feature_vectors/{cls}/prov-gigapath/imagenet'


def new_out(cls):
    return f'{ROOT}/repro/out/{cls}'


def place(cls):
    P = Cls(cls)
    exp, _ = expected()
    for f in FILES:
        path = f'{SRC}/{cls}/{f}'
        if not os.path.isfile(path):
            die(f'missing {path}')
        got = sha256_of(path)
        if got != exp[f'{cls}/{f}']:
            die(f'{path}: sha256 {got} != expected {exp[f"{cls}/{f}"]}')
    print(f'[{cls}] sha256 of the 3 source files match {os.path.relpath(SHA_FILE, ROOT)}')

    feats = np.load(f'{SRC}/{cls}/features.npy')
    ids = np.load(f'{SRC}/{cls}/ids.npy')
    mapping = json.load(open(f'{SRC}/{cls}/ids_2_img_paths.json'))
    uni_map = json.load(open(f'{P.feat}/ids_2_img_paths.json'))
    uni_ids = np.load(f'{P.feat}/ids.npy')
    if feats.shape != (5000, 1536) or feats.dtype != np.float32:
        die(f'features {feats.shape} {feats.dtype}, expected (5000, 1536) float32')
    if not np.isfinite(feats).all():
        die('non-finite features')
    if not np.array_equal(ids, uni_ids):
        die('ids.npy differs from the UNI resize_only ids.npy')
    if mapping != uni_map or list(mapping) != list(uni_map):
        die('id -> path mapping differs from the UNI resize_only mapping')
    print(f'[{cls}] features (5000, 1536) float32 finite | ids.npy and id->path mapping (incl. key order) identical to UNI resize_only')

    dest = dest_dir(cls)
    if os.path.exists(dest) and os.listdir(dest):
        die(f'{dest} already has files; refusing to overwrite')
    os.makedirs(dest, exist_ok=True)
    for f in FILES:
        shutil.copy(f'{SRC}/{cls}/{f}', f'{dest}/{f}')
        if sha256_of(f'{dest}/{f}') != exp[f'{cls}/{f}']:
            die(f'copy of {f} differs from the verified source')
    print(f'[{cls}] copied 3 files to {os.path.relpath(dest, ROOT)} (sha256 re-verified after the copy)')


def seed_spread(cls, P, feats, theirs_ari, theirs_fmi):
    ids2p = json.load(open(f'{dest_dir(cls)}/ids_2_img_paths.json'))
    final = json.load(open(f'{P.ann}/final_clusters.json'))
    p2g = {p: int(g) for g, v in final.items() for p in v}
    true = np.array([p2g[ids2p[str(i)]] for i in range(len(ids2p))])
    k = len(final)
    assert k == P.k_final, (k, P.k_final)
    ari, fmi = [], []
    for s in range(10):
        lab = KMeans(n_clusters=k, random_state=s).fit_predict(feats)
        ari.append(adjusted_rand_score(true, lab))
        fmi.append(fowlkes_mallows_score(true, lab))
        print(f'  seed {s}: ARI {ari[-1]:.4f}', flush=True)
    seed42 = adjusted_rand_score(true, KMeans(n_clusters=k, random_state=42).fit_predict(feats))
    a, f = np.array(ari), np.array(fmi)
    return {'k': k, 'ARI_seed42': seed42, 'ARI_seeds_mean': a.mean(), 'ARI_seeds_sd': a.std(ddof=1), 'ARI_seeds_min': a.min(), 'ARI_seeds_max': a.max(),
            'z_theirs_ARI': (theirs_ari - a.mean()) / a.std(ddof=1), 'theirs_ARI_within_2sd': bool(abs(theirs_ari - a.mean()) <= 2 * a.std(ddof=1)),
            'FMI_seeds_mean': f.mean(), 'FMI_seeds_sd': f.std(ddof=1),
            'z_theirs_FMI': (theirs_fmi - f.mean()) / f.std(ddof=1)}


def row(cls):
    P = Cls(cls)
    out = new_out(cls)
    os.makedirs(out, exist_ok=True)
    targets = [f'{out}/phase3_gigapath_row.csv', f'{out}/phase3_gigapath_eval.csv', f'{out}/phase3_gigapath_eval.json']
    if cls == 'colon_n':
        targets.append(f'{out}/GIGAPATH_NOTE.md')
    clash = [t for t in targets if os.path.exists(t)]
    if clash:
        die(f'refusing to overwrite {clash}')

    scratch = f'{ROOT}/repro/scratch_eval/gigapath/{cls}/eval_results'
    mine_csv = glob.glob(f'{scratch}/*.csv')
    mine_json = glob.glob(f'{scratch}/*.json')
    if len(mine_csv) != 1 or len(mine_json) != 1:
        die(f'expected one evaluation csv and json in {scratch}')
    m = pd.read_csv(mine_csv[0], index_col=0)
    if list(m.index) != [KEY]:
        die(f'scratch evaluation holds {list(m.index)}, expected only {KEY}')
    t = pd.read_csv(glob.glob(P.eval_glob)[0], index_col=0)

    r = {'combo': COMBO}
    for c, s in COLS.items():
        r[f'{s}_theirs'], r[f'{s}_mine'] = float(t.loc[KEY, c]), float(m.loc[KEY, c])
        r[f'{s}_delta'] = r[f'{s}_mine'] - r[f'{s}_theirs']
    for s, bar in BARS.items():
        r[f'{s}_bar'] = bar
        r[f'{s}_pass'] = bool(abs(r[f'{s}_delta']) <= bar)
    r['C2_pass_all4'] = all(r[f'{s}_pass'] for s in BARS)

    print(f'[{cls}] 10-seed KMeans spread on the Colab features (k = {P.k_final}) ...')
    feats = np.load(f'{dest_dir(cls)}/features.npy')
    sp = seed_spread(cls, P, feats, r['ARI_theirs'], r['FMI_theirs'])
    if abs(sp['ARI_seed42'] - r['ARI_mine']) > 1e-9:
        die(f"seed-42 ARI {sp['ARI_seed42']!r} != evaluation ARI {r['ARI_mine']!r}")
    r.update(sp)
    pd.DataFrame([r]).to_csv(targets[0], index=False)
    shutil.copy(mine_csv[0], targets[1])
    shutil.copy(mine_json[0], targets[2])
    print(f'[{cls}] wrote {os.path.relpath(targets[0], ROOT)} (+ eval copies)')
    for s in COLS.values():
        if s in BARS:
            print(f"  {s:4s} theirs {r[f'{s}_theirs']:.5f} mine {r[f'{s}_mine']:.5f} delta {r[f'{s}_delta']:+.5f} bar {BARS[s]} {'PASS' if r[f'{s}_pass'] else 'FAIL'}")
    print(f"  ARI seeds {sp['ARI_seeds_mean']:.4f} +- {sp['ARI_seeds_sd']:.4f} [{sp['ARI_seeds_min']:.4f}, {sp['ARI_seeds_max']:.4f}], theirs z = {sp['z_theirs_ARI']:+.2f}, within 2 sd: {sp['theirs_ARI_within_2sd']}")
    if cls == 'colon_n':
        write_note(targets[3], r)


def write_note(path, r):
    _, prov = expected()
    L = ['# Prov-GigaPath row for colon_n',
         '',
         'New file. No existing colon_n output was changed: `repro/out/phase3_compare.csv` and the other 11-combo outputs still hold 11 combos.',
         'The row below was evaluated in a scratch folder (upstream `eval_results/` was not written) and is also in `phase3_gigapath_row.csv` and `phase3_gigapath_eval.*` next to this note.',
         'The side-by-side summary with colon_aca (11 and 12 combos) is in `repro/REPORT_colon_aca.md`.',
         '',
         '## Provenance and caveat',
         '',
         'These features were extracted on a **Colab T4 GPU** (' + prov[1] + '),',
         'not on the CPU like the other 11 combos (local `venv_repro`: torch 2.14.1+cpu, timm 1.0.30, CPU kernels). Last float digits can therefore differ; the six sha256 lines are in `repro/gigapath_colab_sha256.txt`.',
         '',
         '## prov-gigapath#imagenet, UNI-resize_only annotations, NoReduction, euclidean, KMeans, k = 257',
         '',
         '| metric | theirs | mine | delta | bar | result |',
         '|---|---|---|---|---|---|']
    for s in ('P@1', 'P@5', 'ARI', 'FMI'):
        L.append(f"| {s} | {r[f'{s}_theirs']:.5f} | {r[f'{s}_mine']:.5f} | {r[f'{s}_delta']:+.5f} | {BARS[s]} | {'PASS' if r[f'{s}_pass'] else 'FAIL'} |")
    L += ['',
          f"All four bars: {'PASS' if r['C2_pass_all4'] else 'FAIL'}. KMeans seed spread (seeds 0..9, `n_init` default) on these features: ARI {r['ARI_seeds_mean']:.4f} ± {r['ARI_seeds_sd']:.4f} "
          f"[{r['ARI_seeds_min']:.4f}, {r['ARI_seeds_max']:.4f}]; their ARI is z = {r['z_theirs_ARI']:+.2f} from that mean ({'inside' if r['theirs_ARI_within_2sd'] else 'outside'} 2 sd).",
          '']
    open(path, 'x').write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('place', 'row'))
    ap.add_argument('--cls', choices=CLASSES, required=True)
    a = ap.parse_args()
    {'place': place, 'row': row}[a.step](a.cls)


if __name__ == '__main__':
    main()
