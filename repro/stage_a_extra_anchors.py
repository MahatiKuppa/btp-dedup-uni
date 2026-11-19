"""Phase 1 extra: anchor agreement restricted to clusters identical (as image sets) between my and their initial KMeans."""
import json, warnings
import numpy as np
from scipy.optimize import linear_sum_assignment
warnings.simplefilter('ignore')
UP = 'third_party/LC25000-clean'; A = UP + '/annotations/colon_n/UNI/resize_only'
paths = json.load(open(UP + '/feature_vectors/colon_n/UNI/resize_only/ids_2_img_paths.json')); paths = [paths[str(i)] for i in range(5000)]
p2i = {p: i for i, p in enumerate(paths)}
mine = np.load('repro/out/initial/labels.npy')
anc_m = json.load(open('repro/out/initial/anchors.json'))
rp = json.load(open(A + '/results_processed.json'))
their_anchor = {int(v['cluster_index']): k for k, v in rp.items()}
C = np.zeros((250, 250), int); th = np.zeros(5000, int)
for k, v in rp.items():
    for p in [k] + v['belonging_image_paths'] + v['non_belonging_image_paths']: th[p2i[p]] = v['cluster_index']
np.add.at(C, (mine, th), 1); r, c = linear_sum_assignment(-C)
ident = [(a, b) for a, b in zip(r, c) if C[a, b] == C[a].sum() == C[:, b].sum()]
out = {'identical_clusters': len(ident),
       'anchor_equal_within_identical': int(sum(anc_m[str(a)] == their_anchor[b] for a, b in ident)),
       'their_anchor_in_matched_my_cluster': int(sum(mine[p2i[their_anchor[b]]] == a for a, b in zip(r, c))),
       'their_clusters_with_any_rejection': int(sum(len(v['non_belonging_image_paths']) > 0 for v in rp.values())),
       'their_rejected_images': int(sum(len(v['non_belonging_image_paths']) for v in rp.values()))}
print(out); json.dump(out, open('repro/out/initial/anchor_extra.json', 'w'), indent=1)
