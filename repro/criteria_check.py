"""Evaluate the reproduction criteria in repro/CRITERIA.md against stored results. Prints a table, writes repro/out/criteria_check.json."""
import json, numpy as np
O = 'repro/out/'
p1 = json.load(open(O + 'initial/phase1_metrics.json')); um = json.load(open(O + 'initial/noise_umap_only.json'))
my42_vs_theirs = p1['my_vs_theirs']['ARI']
floors = {'combined (UMAP+KMeans seeds 0..9)': (p1['noise_full_pipeline']['ARI_among_own_mean'], p1['noise_full_pipeline']['ARI_among_own_sd']),
          'UMAP-only (KMeans fixed 42)': (um['ARI_among_own_mean'], um['ARI_among_own_sd']),
          'KMeans-only (UMAP fixed 42)': (p1['noise_kmeans_only_fixed_umap42']['ARI_among_own_mean'], p1['noise_kmeans_only_fixed_umap42']['ARI_among_own_sd'])}
out = {'clusters': {'my_seed42_ARI_vs_their_pkl': my42_vs_theirs, 'floors': {}}, 'features': {}}
for k, (m, s) in floors.items():
    lo, hi = m - 2 * s, m + 2 * s
    out['clusters']['floors'][k] = {'mean': m, 'sd': s, 'lo': lo, 'hi': hi, 'pass': bool(lo <= my42_vs_theirs <= hi)}
out['clusters']['pass_all'] = all(v['pass'] for v in out['clusters']['floors'].values())
r = {}
import csv
for row in csv.DictReader(open(O + 'phase3_uni_resize_only_row.csv')):
    r[row['']] = (float(row['theirs']), float(row['mine']), float(row['delta']))
bars = {'precision@1': 0.002, 'precision@5': 0.002, 'Adjusted Rand Index (ARI)': 0.005, 'Fowlkes-Mallows Index': 0.005}
for m, b in bars.items():
    t, mi, d = r[m]; out['features'][m] = {'theirs': t, 'mine': mi, 'delta': d, 'bar': b, 'pass': bool(abs(d) <= b)}
json.dump(out, open(O + 'criteria_check.json', 'w'), indent=1)
print(f"clusters: my seed-42 ARI vs their pkl = {my42_vs_theirs:.4f}")
for k, v in out['clusters']['floors'].items():
    print(f"  {k}: {v['mean']:.4f} +- {v['sd']:.4f} -> [{v['lo']:.4f}, {v['hi']:.4f}]  {'PASS' if v['pass'] else 'FAIL'}")
for m, v in out['features'].items():
    print(f"features {m}: delta {v['delta']:+.5f} (bar {v['bar']})  {'PASS' if v['pass'] else 'FAIL'}")
