"""Phase 3 extras (own wrapper, upstream code untouched; upstream functions imported).

A. UNI resize_only: k=250 vs k=K (K = number of their final groups: colon_n 257, colon_aca 249) KMeans (seed 42 through upstream metrics; seeds 0..9 label metrics), scored vs their final groups.
B. My 4 old arms (embeddings/*.npy; ordered by my filenames) re-ordered to upstream id order via file-content md5, then
   (i) run through upstream evaluate_clustering() (NoReduction, euclidean, kmeans, k=257, seed 42) and
   (ii) my old saved k=250 labels (results/kmeans_*/labels.npy) re-scored with upstream metric functions.
Outputs: repro/out/[<cls>/]phase3_extras/*.json|csv
"""
import hashlib
import json
import os
import sys
import tempfile
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, completeness_score, fowlkes_mallows_score,
                             homogeneity_score, normalized_mutual_info_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cls import ROOT, UP, parse  # noqa: E402

args, P = parse(__doc__)
sys.path.insert(0, UP)
import evaluate_clustering as EC  # noqa: E402  (upstream script, imported as module)
from source.constants import RANDOM_SEED  # noqa: E402
from source.eval_utils import compute_clustering_metrics  # noqa: E402

OUT = P.extras
os.makedirs(OUT, exist_ok=True)
ANN = P.ann
FD = P.feat
ids2p = json.load(open(f'{FD}/ids_2_img_paths.json'))
paths = [ids2p[str(i)] for i in range(5000)]
tcm, tcv, true_labels, K = EC.get_true_connectivity(ANN, ids2p)
assert K == P.k_final, (K, P.k_final)
KEYS = ['Adjusted Rand Index (ARI)', 'Fowlkes-Mallows Index', 'Normalized Mutual Information (NMI)', 'Homogeneity', 'Completeness']


def full_metrics(pred_labels):
    pcv = EC.get_predicted_connectivity(pred_labels, ids2p)
    m = compute_clustering_metrics(tcv, pcv, true_labels, pred_labels)
    return {k: float(m[k]) for k in KEYS}


def label_metrics(lab):
    return dict(zip(KEYS, [adjusted_rand_score(true_labels, lab), fowlkes_mallows_score(true_labels, lab),
                           normalized_mutual_info_score(true_labels, lab), homogeneity_score(true_labels, lab),
                           completeness_score(true_labels, lab)]))


# ------------------------------------------------------------------ A. k=250 vs 257
feats = np.load(f'{FD}/features.npy')
A = {}
for k in (() if os.path.exists(f'{OUT}/k250_vs_k{K}_uni.json') else (K, 250)):
    lab42 = KMeans(n_clusters=k, random_state=RANDOM_SEED).fit_predict(feats)
    seeds = [label_metrics(KMeans(n_clusters=k, random_state=s).fit_predict(feats)) for s in range(10)]
    df = pd.DataFrame(seeds)
    A[k] = {'seed42_upstream_metrics': full_metrics(lab42),
            'seeds0_9_mean': df.mean().to_dict(), 'seeds0_9_sd': df.std(ddof=1).to_dict()}
    print('A k =', k, {m: round(v, 4) for m, v in A[k]['seed42_upstream_metrics'].items()},
          '| seeds mean ARI %.4f sd %.4f' % (df.iloc[:, 0].mean(), df.iloc[:, 0].std(ddof=1)), flush=True)
if A:
    json.dump(A, open(f'{OUT}/k250_vs_k{K}_uni.json', 'w'), indent=1)

# ------------------------------------------------------------------ B. old arms
def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


mine_names = [l.strip() for l in open(f'{P.emb}/paths.txt')]
assert len(mine_names) == 5000 and mine_names == [l.strip() for l in open(f'{P.emb}/resnet50_paths.txt')]
their_md5 = {}
for p in paths:
    their_md5.setdefault(md5(f'{UP}/{p}'), []).append(p)
old_md5 = {}
for i, n in enumerate(mine_names):
    old_md5.setdefault(md5(f'{ROOT}/{n}'), []).append(i)
# 454 files are byte-identical duplicates inside colon_n, so md5 is not 1:1. Identical content => identical embedding and identical old
# KMeans assignment, so any bijection inside a duplicate group yields the same reordered matrix. Pair sequentially and check counts.
assert set(their_md5) == set(old_md5) and all(len(their_md5[h]) == len(old_md5[h]) for h in their_md5), 'md5 groups differ between old data and upstream data'
n_dup_groups = sum(len(v) > 1 for v in their_md5.values())
print('md5 groups:', len(their_md5), '| groups with duplicates:', n_dup_groups)
perm_d = {}
for h, ps in their_md5.items():
    for p, i in zip(sorted(ps), sorted(old_md5[h])):
        perm_d[p] = i
perm = np.array([perm_d[p] for p in paths])
assert len(set(perm)) == 5000
csv = pd.read_csv(f'{P.emb}/my_filename_to_group_id.csv')
csvmap = dict(zip(csv.my_filename, csv.their_filename))
agree = sum(f"{P.img_prefix}{csvmap[os.path.basename(mine_names[perm[i]])]}" == paths[i] for i in range(5000))
print('reordering agrees with old csv name mapping for', agree, '/ 5000 (rest = duplicate-group ambiguity)')

arms = {'uni1_full': 'kmeans_uni', 'uni1_pca128': 'kmeans_uni_pca128', 'resnet50_full': 'kmeans_resnet', 'resnet50_pca128': 'kmeans_resnet_pca128'}
B = {}
for emb, res_dir in arms.items():
    X = np.load(f'{P.emb}/{emb}.npy')[perm]
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as td:
        np.save(f'{td}/features.npy', X.astype(np.float32))
        json.dump(ids2p, open(f'{td}/ids_2_img_paths.json', 'w'))
        m = EC.evaluate_clustering(td, ANN, 'NoReduction', 'kmeans', 'euclidean', False)   # upstream pipeline, k=257, seed 42
    old_lab = np.load(f'{P.res}/{res_dir}/labels.npy')[perm]                       # my old k=250 labels
    B[emb] = {f'upstream_pipeline_k{K}': {k: float(m[k]) for k in KEYS + ['precision@1', 'precision@5']},
              'old_k250_labels_rescored': full_metrics(old_lab)}
    print('B', emb, {k: round(v, 4) for k, v in B[emb][f'upstream_pipeline_k{K}'].items()}, flush=True)
json.dump(B, open(f'{OUT}/old_arms_rescored.json', 'w'), indent=1)

rows = [{'arm': a, 'setting': s, **v} for a, d in B.items() for s, v in d.items()]
pd.DataFrame(rows).to_csv(f'{OUT}/old_arms_rescored.csv', index=False)
print('saved', OUT)
