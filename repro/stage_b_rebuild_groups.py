"""Phase 2: rebuild the pre-merge cluster set from upstream's recorded decisions (as notebook 2 cell 30) and diff against final_clusters.json.
Read-only on upstream annotations. Writes repro/out/phase2/.
"""
import json
import os
from collections import Counter

import numpy as np

ROOT = os.path.expanduser('~/btp_dedup_uni')
ANN = f'{ROOT}/third_party/LC25000-clean/annotations/colon_n/UNI/resize_only'
OUT = f'{ROOT}/repro/out/phase2'
os.makedirs(OUT, exist_ok=True)
NUM_CLASS_PROTOTYPES = 250  # source.constants

rp = json.load(open(f'{ANN}/results_processed.json'))
pure_rej = json.load(open(f'{ANN}/pure_rejected_clusters.json'))
final = json.load(open(f'{ANN}/final_clusters.json'))

# ---- reconstruct pre-merge set exactly as the notebook: accepted -> cluster_index, rejected -> 250 + id
pre = {}
initial = {}   # the initial KMeans clusters (accepted + non-belonging), for comparison
for anchor, v in rp.items():
    cid = int(v['cluster_index'])
    pre[cid] = [anchor] + v['belonging_image_paths']
    initial[cid] = [anchor] + v['belonging_image_paths'] + v['non_belonging_image_paths']
for cid, imgs in pure_rej.items():
    pre[NUM_CLASS_PROTOTYPES + int(cid)] = list(imgs)

allimgs = [p for v in pre.values() for p in v]
assert len(allimgs) == 5000 and len(set(allimgs)) == 5000, (len(allimgs), len(set(allimgs)))
assert sorted(set(p for v in initial.values() for p in v)) == sorted(allimgs)
print(f'pre-merge clusters: {len(pre)} ({sum(k < 250 for k in pre)} accepted, {sum(k >= 250 for k in pre)} rejected-pure); every image exactly once: OK')

fin_imgs = [p for v in final.values() for p in v]
assert len(fin_imgs) == 5000 and set(fin_imgs) == set(allimgs)
print(f'final groups: {len(final)}')

# ---- map each pre-merge cluster into final groups
img2final = {p: int(g) for g, v in final.items() for p in v}
split_pre = {}   # pre-merge clusters whose images land in >1 final group (would violate "merge only" assumption)
pre2final = {}
for cid, imgs in pre.items():
    gs = Counter(img2final[p] for p in imgs)
    if len(gs) > 1:
        split_pre[cid] = dict(gs)
    pre2final[cid] = gs.most_common(1)[0][0]
print('pre-merge clusters split across final groups (should be 0 if final = pure unions):', len(split_pre))

members = {}
for cid, g in pre2final.items():
    members.setdefault(g, []).append(cid)
impure = [g for g, cids in members.items()
          if sum(len(pre[c]) for c in cids) != len(final[str(g)])]
print('final groups that are NOT an exact union of pre-merge clusters:', len(impure))

# ---- merges
kind = Counter()
merged = []
for g, cids in sorted(members.items()):
    if len(cids) > 1:
        a = sorted(c for c in cids if c < 250); r = sorted(c for c in cids if c >= 250)
        kind[f'{len(a)}acc+{len(r)}rej'] += 1
        merged.append({'final_group': g, 'accepted_clusters': a, 'rejected_clusters': [c - 250 for c in r],
                       'sizes': [len(pre[c]) for c in sorted(cids)], 'final_size': len(final[str(g)])})
n_merges = len(pre) - len(final)
print(f'merge operations implied: {n_merges}; final groups composed of >1 pre-merge cluster: {len(merged)}')
print('composition of merged groups:', dict(kind))

# ---- fractions
fs = {g: set(v) for g, v in final.items()}
init_sets = [frozenset(v) for v in initial.values()]
pre_sets = [frozenset(v) for v in pre.values()]
eq_init = sum(frozenset(v) in set(init_sets) for v in fs.values())
eq_pre = sum(frozenset(v) in set(pre_sets) for v in fs.values())
print(f'final groups identical to ONE initial KMeans cluster (no rejection, no merge): {eq_init}/{len(final)} = {eq_init/len(final):.3f}')
print(f'final groups identical to ONE pre-merge cluster (rejections allowed, no merge): {eq_pre}/{len(final)} = {eq_pre/len(final):.3f}')

# ---- rejection rate per initial cluster
rej = {}
for anchor, v in rp.items():
    n = 1 + len(v['belonging_image_paths']) + len(v['non_belonging_image_paths'])
    rej[int(v['cluster_index'])] = (len(v['non_belonging_image_paths']), n)
rates = np.array([a / b for a, b in rej.values()])
nz = rates[rates > 0]
print(f'initial clusters with >=1 rejected image: {len(nz)}/250; rejected images: {sum(a for a, _ in rej.values())}/5000 '
      f'= {sum(a for a, _ in rej.values())/5000:.4f}')
print(f'per-cluster rejection rate: mean(all)={rates.mean():.4f} mean(nonzero)={nz.mean():.3f} max={rates.max():.3f}; '
      f'rates>0.5: {(rates > 0.5).sum()}, ==1.0 (whole cluster rejected): {(rates >= 1).sum()}')
print('rejected pure clusters sizes:', sorted(len(v) for v in pure_rej.values()))
# where do rejected pure clusters end up?
rej_alone = sum(1 for g, cids in members.items() if len(cids) == 1 and cids[0] >= 250)
rej_merged_acc = sum(1 for g, cids in members.items() if any(c >= 250 for c in cids) and any(c < 250 for c in cids))
rej_merged_rej = sum(1 for g, cids in members.items() if len(cids) > 1 and all(c >= 250 for c in cids))
print(f'final groups that are a rejected-pure cluster alone: {rej_alone}; merged into accepted cluster(s): {rej_merged_acc}; merged only with other rejected: {rej_merged_rej}')


def sz(d):
    return sorted((len(v) for v in d.values()), reverse=True)


res = {'n_pre_merge': len(pre), 'n_accepted': 250, 'n_rejected_pure': len(pure_rej), 'n_final': len(final), 'n_merges_implied': n_merges,
       'final_groups_equal_single_initial_cluster': eq_init, 'final_groups_equal_single_pre_merge_cluster': eq_pre,
       'split_pre_merge_clusters': split_pre, 'final_groups_not_exact_union': impure, 'merged_groups': merged,
       'merge_composition': dict(kind), 'rejected_images': sum(a for a, _ in rej.values()),
       'clusters_with_rejection': int(len(nz)), 'rejection_rate_per_initial_cluster': {str(k): v for k, v in rej.items()},
       'rejected_pure_alone_final': rej_alone, 'rejected_pure_merged_with_accepted': rej_merged_acc,
       'sizes_final': sz(final), 'sizes_initial250': sz(initial), 'sizes_pre_merge': sz(pre)}
json.dump(res, open(f'{OUT}/phase2_metrics.json', 'w'), indent=1)
json.dump({str(k): v for k, v in pre.items()}, open(f'{OUT}/pre_merge_clusters.json', 'w'))
print('saved', OUT)
