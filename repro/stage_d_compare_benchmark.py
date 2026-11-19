"""Phase 3: my evaluate_clustering CSV vs upstream's committed colon_n CSV, per combo, with the CRITERIA.md C2 bars. Re-run after prov-gigapath arrives."""
import glob, sys
import pandas as pd
ROOT = '/home/mahati/btp_dedup_uni'
mine_dir = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/repro/scratch_eval'
t = pd.read_csv(glob.glob(f'{ROOT}/third_party/LC25000-clean/eval_results/*colon_n*.csv')[0], index_col=0)
m = pd.read_csv(glob.glob(f'{mine_dir}/eval_results/*.csv')[0], index_col=0)
cols = {'Adjusted Rand Index (ARI)': 'ARI', 'Fowlkes-Mallows Index': 'FMI', 'Normalized Mutual Information (NMI)': 'NMI',
        'Homogeneity': 'Homog', 'Completeness': 'Compl', 'precision@1': 'P@1', 'precision@5': 'P@5'}
bars = {'ARI': 0.005, 'FMI': 0.005, 'P@1': 0.002, 'P@5': 0.002}
assert set(m.index) <= set(t.index), set(m.index) - set(t.index)
rows, long = [], []
for k in m.index:
    r = {'combo': k.replace('#euclidean#NoReduction#kmeans', '')}
    ok = True
    for c, s in cols.items():
        d = float(m.loc[k, c]) - float(t.loc[k, c])
        r[f'{s}_theirs'], r[f'{s}_mine'], r[f'{s}_delta'] = float(t.loc[k, c]), float(m.loc[k, c]), d
        if s in bars:
            ok &= abs(d) <= bars[s]
    r['C2_pass_all4'] = ok
    rows.append(r)
df = pd.DataFrame(rows).sort_values('combo')
df.to_csv(f'{ROOT}/repro/out/phase3_compare.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50)
print(df[['combo'] + [f'{s}_delta' for s in cols.values()] + ['C2_pass_all4']].round(4).to_string(index=False))
print('\nmine rows:', len(m), '| their rows:', len(t), '| missing from mine:', sorted(set(t.index) - set(m.index)))
print('\ntheirs vs mine (ARI):'); print(df[['combo', 'ARI_theirs', 'ARI_mine']].round(4).to_string(index=False))
print('\nmean |delta|:', {s: round(df[f'{s}_delta'].abs().mean(), 4) for s in cols.values()})
