"""Phase 1: initial clustering exactly as notebook 2 (automated part), compared with the upstream annotation.

Run from anywhere:  venv_repro/bin/python repro/stage_a_initial_clustering.py
Reads (read-only): upstream feature_vectors/ + annotations/colon_n/UNI/resize_only/.
Writes only to repro/out/initial/.
"""
import json
import os
import pickle
import sys
import warnings

import numpy as np
import umap
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances

ROOT = os.path.expanduser('~/btp_dedup_uni')
UP = f'{ROOT}/third_party/LC25000-clean'
sys.path.insert(0, UP)
from source.constants import NUM_CLASS_PROTOTYPES, RANDOM_SEED  # noqa: E402
from source.eval_utils import (get_clustering_centroids, get_clustering_labels,  # noqa: E402
                               reduce_feature_dimensionality)

FEAT = f'{UP}/feature_vectors/colon_n/UNI/resize_only'
ANN = f'{UP}/annotations/colon_n/UNI/resize_only'
OUT = f'{ROOT}/repro/out/initial'
os.makedirs(OUT, exist_ok=True)


def sizes(labels):
    return np.sort(np.bincount(labels))[::-1]


def hungarian_agreement(a, b):
    """Contingency of two labelings, optimal one-to-one cluster matching."""
    ka, kb = a.max() + 1, b.max() + 1
    C = np.zeros((ka, kb), dtype=np.int64)
    np.add.at(C, (a, b), 1)
    r, c = linear_sum_assignment(-C)
    matched = C[r, c]
    sa, sb = C.sum(1), C.sum(0)
    identical = int(np.sum((matched == sa[r]) & (matched == sb[c])))
    return {'frac_images_in_matched_cluster': float(matched.sum() / len(a)),
            'n_identical_clusters': identical}


def pair(a, b):
    return {'ARI': float(adjusted_rand_score(a, b)), 'AMI': float(adjusted_mutual_info_score(a, b)),
            **hungarian_agreement(a, b)}


def anchors_of(feats_reduced, labels):
    cents = get_clustering_centroids(feats_reduced, labels)
    anchors = {}
    for lab in range(len(cents)):
        idx = np.where(labels == lab)[0]
        d = euclidean_distances([cents[lab]], feats_reduced[idx])
        anchors[lab] = int(idx[np.argmin(d)])
    return cents, anchors


def umap_kmeans(features, umap_seed, km_seed):
    red = umap.UMAP(n_components=8, random_state=umap_seed).fit_transform(features)
    lab = KMeans(n_clusters=NUM_CLASS_PROTOTYPES, random_state=km_seed).fit_predict(red)
    return red, lab


# ---------------------------------------------------------------- load
features = np.load(f'{FEAT}/features.npy')
ids2paths = json.load(open(f'{FEAT}/ids_2_img_paths.json'))
paths = [ids2paths[str(i)] for i in range(len(ids2paths))]
path2idx = {p: i for i, p in enumerate(paths)}
assert features.shape == (5000, 1024) and len(set(paths)) == 5000
print('features', features.shape, features.dtype)

# ---------------------------------------------------------------- their pkl + their recorded initial clustering
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    with open(f'{ANN}/distance_metric=euclidean#dimensionality_reduction=UMAP-8#clustering=kmeans.pkl', 'rb') as f:
        theirs = pickle.load(f)
their_sklearn = next((str(x.message).split('from version ')[1].split(' when')[0] for x in w
                      if 'InconsistentVersionWarning' in type(x.message).__name__), 'unknown')
their_labels = np.asarray(theirs.labels_)
print(f'their pkl: sklearn {their_sklearn} (mine: see env), n_init={theirs.n_init}, n_features_in_={theirs.n_features_in_}, '
      f'n_clusters={theirs.n_clusters}, init={theirs.init}, n_iter_={theirs.n_iter_}, inertia={theirs.inertia_:.4f}')
print('their cluster sizes (sorted desc): top10', sizes(their_labels)[:10], 'bottom10', sizes(their_labels)[-10:])

rp = json.load(open(f'{ANN}/results_processed.json'))
rp_labels = np.full(5000, -1)
for anchor, d in rp.items():
    for p in [anchor] + d['belonging_image_paths'] + d['non_belonging_image_paths']:
        rp_labels[path2idx[p]] = int(d['cluster_index'])
assert (rp_labels >= 0).all()
order_ok = bool((rp_labels == their_labels).all())
print('their pkl labels (in MY id order) == labels recorded in results_processed.json (by path):', order_ok)
their_anchor_paths = set(rp.keys())

# ---------------------------------------------------------------- my run, exactly as notebook 2
red = reduce_feature_dimensionality(features, 'UMAP-8')
my_labels, my_model = get_clustering_labels(red, NUM_CLASS_PROTOTYPES, 'kmeans', RANDOM_SEED, return_model=True)
cents, anchors = anchors_of(red, my_labels)
my_anchor_paths = {paths[i] for i in anchors.values()}
np.save(f'{OUT}/labels.npy', my_labels)
np.save(f'{OUT}/centers.npy', cents)
np.save(f'{OUT}/umap8.npy', red)
json.dump({str(k): paths[v] for k, v in anchors.items()}, open(f'{OUT}/anchors.json', 'w'), indent=1)
print(f'mine: n_init={my_model.n_init}, n_iter_={my_model.n_iter_}, inertia={my_model.inertia_:.4f}')

res = {'their_sklearn_version': their_sklearn, 'their_n_init': str(theirs.n_init),
       'their_n_features_in': int(theirs.n_features_in_), 'order_check_pkl_vs_results_processed': order_ok,
       'my_n_init': str(my_model.n_init), 'my_vs_theirs': pair(my_labels, their_labels),
       'anchors_equal_to_theirs': len(my_anchor_paths & their_anchor_paths),
       'sizes_mine_sorted': sizes(my_labels).tolist(), 'sizes_theirs_sorted': sizes(their_labels).tolist()}
print('MINE vs THEIRS', res['my_vs_theirs'], '| anchors equal:', res['anchors_equal_to_theirs'], '/ 250')


def quant(s):
    return {k: float(v) for k, v in zip(['min', 'q25', 'median', 'q75', 'max', 'mean', 'sd'],
                                        [s.min(), *np.percentile(s, [25, 50, 75]), s.max(), s.mean(), s.std()])}


print('size stats mine  ', quant(sizes(my_labels)))
print('size stats theirs', quant(sizes(their_labels)))
res['size_stats_mine'], res['size_stats_theirs'] = quant(sizes(my_labels)), quant(sizes(their_labels))
res['sorted_size_L1'] = int(np.abs(sizes(my_labels) - sizes(their_labels)).sum())

# ---------------------------------------------------------------- noise floor
# (a) determinism: repeat seed-42 pipeline
_, again = umap_kmeans(features, RANDOM_SEED, RANDOM_SEED)
res['rerun_seed42_vs_mine'] = pair(again, my_labels)
print('re-run seed 42 vs mine (determinism):', res['rerun_seed42_vs_mine'])

# (b) full pipeline, UMAP seed = KMeans seed = s
runs = {}
for s in range(10):
    _, runs[s] = umap_kmeans(features, s, s)
    np.save(f'{OUT}/labels_seed{s}.npy', runs[s])
    print('seed', s, 'done', flush=True)
own = [adjusted_rand_score(runs[i], runs[j]) for i in range(10) for j in range(i + 1, 10)]
vs_theirs = [adjusted_rand_score(runs[s], their_labels) for s in range(10)]
vs_mine42 = [adjusted_rand_score(runs[s], my_labels) for s in range(10)]
res['noise_full_pipeline'] = {'ARI_among_own_mean': float(np.mean(own)), 'ARI_among_own_sd': float(np.std(own)),
                              'ARI_vs_theirs_mean': float(np.mean(vs_theirs)), 'ARI_vs_theirs_sd': float(np.std(vs_theirs)),
                              'ARI_vs_my_seed42_mean': float(np.mean(vs_mine42)), 'ARI_vs_my_seed42_sd': float(np.std(vs_mine42)),
                              'ARI_vs_theirs_each': vs_theirs}
print('NOISE full pipeline', {k: v for k, v in res['noise_full_pipeline'].items() if k != 'ARI_vs_theirs_each'})

# (c) KMeans-only noise: fixed UMAP(42) embedding, KMeans seeds 0..9 (separates UMAP vs KMeans variability)
km = [KMeans(n_clusters=NUM_CLASS_PROTOTYPES, random_state=s).fit_predict(red) for s in range(10)]
own_km = [adjusted_rand_score(km[i], km[j]) for i in range(10) for j in range(i + 1, 10)]
km_vs_theirs = [adjusted_rand_score(k, their_labels) for k in km]
res['noise_kmeans_only_fixed_umap42'] = {'ARI_among_own_mean': float(np.mean(own_km)), 'ARI_among_own_sd': float(np.std(own_km)),
                                         'ARI_vs_theirs_mean': float(np.mean(km_vs_theirs)),
                                         'ARI_vs_theirs_sd': float(np.std(km_vs_theirs))}
print('NOISE kmeans-only', res['noise_kmeans_only_fixed_umap42'])

json.dump(res, open(f'{OUT}/phase1_metrics.json', 'w'), indent=1)
print('saved', f'{OUT}/phase1_metrics.json')
