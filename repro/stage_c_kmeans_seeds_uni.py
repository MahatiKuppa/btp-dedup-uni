"""KMeans seed / n_init variation at k=257 on UNI resize_only (NoReduction, euclidean), scored against upstream final_clusters.json.
Sanity: the seed-42 / n_init='auto' ARI must equal the number evaluate_clustering.py produced in repro/scratch_eval_uni_resize.
Label-based metrics (ARI, FMI, NMI, homogeneity, completeness) are computed with the same sklearn functions upstream's compute_clustering_metrics uses.
"""
import json, os, sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score,
                             homogeneity_score, completeness_score)
ROOT = os.path.expanduser('~/btp_dedup_uni'); UP = f'{ROOT}/third_party/LC25000-clean'
FD = f'{UP}/feature_vectors/colon_n/UNI/resize_only'
feats = np.load(f'{FD}/features.npy')
ids = json.load(open(f'{FD}/ids_2_img_paths.json'))
final = json.load(open(f'{UP}/annotations/colon_n/UNI/resize_only/final_clusters.json'))
p2g = {p: int(g) for g, v in final.items() for p in v}
true = np.array([p2g[ids[str(i)]] for i in range(5000)]); K = len(final); assert K == 257
THEIRS = {'ARI': 0.63765, 'FMI': 0.64027}   # committed eval_results row (5 d.p. from phase3_uni_resize_only_row.csv)

def score(lab):
    return {'ARI': adjusted_rand_score(true, lab), 'FMI': fowlkes_mallows_score(true, lab), 'NMI': normalized_mutual_info_score(true, lab),
            'Homogeneity': homogeneity_score(true, lab), 'Completeness': completeness_score(true, lab)}

res = {}
for n_init in ['auto', 10]:
    rows = {}
    for s in list(range(10)) + [42]:
        lab = KMeans(n_clusters=K, random_state=s, n_init=n_init).fit_predict(feats)
        rows[s] = score(lab); print(n_init, s, {k: round(v, 5) for k, v in rows[s].items()}, flush=True)
    res[str(n_init)] = rows
summ = {}
for n_init, rows in res.items():
    d = {}
    for m in ['ARI', 'FMI', 'NMI', 'Homogeneity', 'Completeness']:
        v = np.array([rows[str(s)][m] if str(s) in rows else rows[s][m] for s in range(10)])
        d[m] = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)), 'min': float(v.min()), 'max': float(v.max())}
    for m, t in THEIRS.items():
        v = np.array([rows[s][m] for s in range(10)])
        d[m].update({'theirs': t, 'z_theirs': float((t - v.mean()) / v.std(ddof=1)), 'n_runs_ge_theirs': int((v >= t).sum()),
                     'n_runs_within_0.005_of_theirs': int((abs(v - t) <= 0.005).sum()), 'seed42': float(rows[42][m])})
    summ[n_init] = d
    print('\n== n_init =', n_init, '(seeds 0..9, sd with ddof=1)')
    for m in ['ARI', 'FMI']:
        x = d[m]; print(f"{m}: {x['mean']:.4f} +- {x['sd']:.4f} [{x['min']:.4f},{x['max']:.4f}] theirs {x['theirs']} z={x['z_theirs']:+.2f} "
                       f"runs>=theirs {x['n_runs_ge_theirs']}/10 within0.005 {x['n_runs_within_0.005_of_theirs']}/10 seed42 {x['seed42']:.5f}")
json.dump({'per_run': {k: {str(s): r for s, r in v.items()} for k, v in res.items()}, 'summary': summ}, open(f'{ROOT}/repro/out/kmeans_seeds_uni_k257.json', 'w'), indent=1)
