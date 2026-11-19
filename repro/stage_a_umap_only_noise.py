"""Phase 1 extra: UMAP-only noise floor. UMAP seeds 0..9, KMeans seed fixed at 42 (complement of the KMeans-only floor in stage_a)."""
import json, os, sys
import numpy as np, umap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
ROOT = os.path.expanduser('~/btp_dedup_uni'); UP = f'{ROOT}/third_party/LC25000-clean'
sys.path.insert(0, UP)
from source.constants import NUM_CLASS_PROTOTYPES, RANDOM_SEED
feats = np.load(f'{UP}/feature_vectors/colon_n/UNI/resize_only/features.npy')
ids = json.load(open(f'{UP}/feature_vectors/colon_n/UNI/resize_only/ids_2_img_paths.json'))
import pickle, warnings; warnings.simplefilter('ignore')
theirs = pickle.load(open(f'{UP}/annotations/colon_n/UNI/resize_only/distance_metric=euclidean#dimensionality_reduction=UMAP-8#clustering=kmeans.pkl', 'rb')).labels_
mine42 = np.load(f'{ROOT}/repro/out/initial/labels.npy')
runs = []
for s in range(10):
    red = umap.UMAP(n_components=8, random_state=s).fit_transform(feats)
    runs.append(KMeans(n_clusters=NUM_CLASS_PROTOTYPES, random_state=RANDOM_SEED).fit_predict(red)); print('seed', s, flush=True)
own = [ari(runs[i], runs[j]) for i in range(10) for j in range(i + 1, 10)]
vt = [ari(r, theirs) for r in runs]; vm = [ari(r, mine42) for r in runs]
res = {'ARI_among_own_mean': np.mean(own), 'ARI_among_own_sd': np.std(own), 'ARI_vs_theirs_mean': np.mean(vt), 'ARI_vs_theirs_sd': np.std(vt),
       'ARI_vs_my_seed42_mean': np.mean(vm), 'ARI_vs_my_seed42_sd': np.std(vm), 'ARI_vs_theirs_each': vt}
res = {k: (float(v) if not isinstance(v, list) else v) for k, v in res.items()}
print(res); json.dump(res, open(f'{ROOT}/repro/out/initial/noise_umap_only.json', 'w'), indent=1)
