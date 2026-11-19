"""Per-combo KMeans seed noise (k=257, NoReduction, euclidean, n_init default 'auto'), seeds 0..9, vs their 257 groups.
Puts each upstream-committed ARI/FMI in context: z = (theirs - mean) / sd."""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari, fowlkes_mallows_score as fmi
ROOT = os.path.expanduser('~/btp_dedup_uni'); UP = f'{ROOT}/third_party/LC25000-clean'
final = json.load(open(f'{UP}/annotations/colon_n/UNI/resize_only/final_clusters.json')); p2g = {p: int(g) for g, v in final.items() for p in v}
t = pd.read_csv(glob.glob(f'{UP}/eval_results/*colon_n*.csv')[0], index_col=0)
rows = []
for d in sorted(glob.glob(f'{UP}/feature_vectors/colon_n/*/*/features.npy')):
    d = os.path.dirname(d); ext, norm = d.split('/')[-2:]
    key = f'{ext}#{norm}#euclidean#NoReduction#kmeans'
    X = np.load(f'{d}/features.npy'); ids = json.load(open(f'{d}/ids_2_img_paths.json'))
    y = np.array([p2g[ids[str(i)]] for i in range(5000)])
    a, f = [], []
    for s in range(10):
        lab = KMeans(n_clusters=257, random_state=s).fit_predict(X); a.append(ari(y, lab)); f.append(fmi(y, lab))
    a42 = ari(y, KMeans(n_clusters=257, random_state=42).fit_predict(X))
    ta = float(t.loc[key, 'Adjusted Rand Index (ARI)'])
    r = {'combo': f'{ext}#{norm}', 'ARI_theirs': ta, 'ARI_mine_seed42': a42, 'ARI_seeds_mean': np.mean(a), 'ARI_seeds_sd': np.std(a, ddof=1),
         'ARI_seeds_min': min(a), 'ARI_seeds_max': max(a), 'z_theirs': (ta - np.mean(a)) / np.std(a, ddof=1), 'z_seed42': (a42 - np.mean(a)) / np.std(a, ddof=1),
         'theirs_inside_min_max': bool(min(a) <= ta <= max(a)), 'theirs_within_2sd': bool(abs(ta - np.mean(a)) <= 2 * np.std(a, ddof=1))}
    rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
pd.DataFrame(rows).to_csv(f'{ROOT}/repro/out/phase3_kmeans_seed_noise_all.csv', index=False)
