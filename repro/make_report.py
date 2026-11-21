"""Size-distribution figures and repro/REPORT_<cls>.md.

    venv_repro/bin/python repro/make_report.py --figs colon_n      # figures only, reads existing colon_n outputs
    venv_repro/bin/python repro/make_report.py --cls colon_aca     # figures + REPORT_colon_aca.md

Everything in the colon_n columns is read from existing colon_n outputs. The one exception is the cluster
sizes of "my benchmark KMeans run" (UNI resize_only, k = number of their final groups, seed 42, n_init default),
which were never stored: they are re-derived in memory from the existing feature vectors and the figure is only
written if the resulting ARI equals the stored value in phase3_uni_resize_only_row.csv.
"""
import argparse
import csv
import glob
import importlib
import json
import os
import platform
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cls import ROOT, UP, Cls  # noqa: E402

HASH_BEFORE = f'{ROOT}/markers/colon_n_repro_hashes_before.txt'
BENCH_KEY = 'UNI#resize_only#euclidean#NoReduction#kmeans'
COMBO_GIGA = 'prov-gigapath#imagenet'


def jl(path):
    return json.load(open(path))


def f4(x, n=4):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{n}f}'


def md(header, rows):
    return ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]


# --------------------------------------------------------------------------- figures
def benchmark_kmeans_sizes(P):
    """Sizes of KMeans(k=K, seed 42, default n_init) on UNI resize_only; verified against the stored benchmark ARI."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    feats = np.load(f'{P.feat}/features.npy')
    ids = jl(f'{P.feat}/ids_2_img_paths.json')
    final = jl(f'{P.ann}/final_clusters.json')
    p2g = {p: int(g) for g, v in final.items() for p in v}
    truth = np.array([p2g[ids[str(i)]] for i in range(len(ids))])
    lab = KMeans(n_clusters=len(final), random_state=42).fit_predict(feats)
    ari = adjusted_rand_score(truth, lab)
    stored = pd.read_csv(f'{P.out}/phase3_uni_resize_only_row.csv', index_col=0).loc['Adjusted Rand Index (ARI)', 'mine']
    if abs(ari - stored) > 1e-9:
        return None, f'ARI {ari!r} != stored {stored!r}'
    return np.sort(np.bincount(lab))[::-1], f'ARI {ari:.5f} equals stored benchmark value'


def size_figure(sizes, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sizes = np.asarray(sizes)
    ink, grid, series = '#1f2328', '#d0d7de', '#2a6f97'
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.4), dpi=150)
    bins = np.arange(sizes.min(), sizes.max() + 2) - 0.5
    a.hist(sizes, bins=bins, color=series)
    a.set_xlabel('cluster size (images)', color=ink)
    a.set_ylabel('number of clusters', color=ink)
    a.set_title('size histogram', fontsize=10, color=ink)
    b.plot(np.arange(1, len(sizes) + 1), np.sort(sizes)[::-1], color=series, lw=1.8)
    b.set_xlabel('cluster rank (largest first)', color=ink)
    b.set_ylabel('cluster size (images)', color=ink)
    b.set_title('sorted sizes', fontsize=10, color=ink)
    for ax in (a, b):
        ax.grid(True, color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    stats = f'n={len(sizes)}  min={sizes.min()}  median={np.median(sizes):.0f}  max={sizes.max()}  mean={sizes.mean():.2f}  sd={sizes.std():.2f}'
    fig.suptitle(f'{title}\n{stats}', fontsize=10, color=ink)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor='white')
    plt.close(fig)


def make_figures(cls):
    P = Cls(cls)
    figdir = f'{ROOT}/repro/figs/{cls}'
    p1 = jl(f'{P.initial}/phase1_metrics.json')
    p2 = jl(f'{P.phase2}/phase2_metrics.json')
    k = len(jl(f'{P.ann}/final_clusters.json'))
    bench, note = benchmark_kmeans_sizes(P)
    items = [('sizes_their_final.png', f'{cls}: their final groups (k={k})', p2['sizes_final']),
             ('sizes_their_initial250.png', f'{cls}: their initial KMeans clusters (k=250, UMAP-{P.their_dim})', p2['sizes_initial250']),
             ('sizes_my_initial250.png', f'{cls}: my initial KMeans clusters (k=250, UMAP-{P.their_dim}, seed 42)', p1['sizes_mine_sorted'])]
    if bench is not None:
        items.append(('sizes_my_benchmark_kmeans.png', f'{cls}: my benchmark KMeans run (UNI resize_only, k={k}, seed 42)', bench))
    written = []
    for name, title, sizes in items:
        if not os.path.exists(f'{figdir}/{name}'):   # existing figures are never rewritten
            size_figure(sizes, title, f'{figdir}/{name}')
        written.append(f'repro/figs/{cls}/{name}')
    print(f'[figs] {cls}: {len(written)} figures | benchmark sizes: {note}')
    return written, note


# --------------------------------------------------------------------------- data
def load(cls, umap_dim=None):
    P = Cls(cls, umap_dim)
    d = {'P': P}
    d['p1'] = jl(f'{P.initial}/phase1_metrics.json')
    d['um'] = jl(f'{P.initial}/noise_umap_only.json')
    d['anc'] = jl(f'{P.initial}/anchor_extra.json')
    crit = 'criteria_check.json' if P.umap_dim == P.their_dim else f'criteria_check_umap{P.umap_dim}.json'
    d['crit'] = jl(f'{P.out}/{crit}')
    if umap_dim is None:
        d['p2'] = jl(f'{P.phase2}/phase2_metrics.json')
        d['cmp'] = pd.read_csv(f'{P.out}/phase3_compare.csv').set_index('combo')
        d['seeds'] = jl(f'{P.out}/kmeans_seeds_uni_k{P.k_final}.json')['summary']
        d['noise_all'] = pd.read_csv(f'{P.out}/phase3_kmeans_seed_noise_all.csv').set_index('combo')
        d['k'] = jl(f'{P.extras}/k250_vs_k{P.k_final}_uni.json')
        d['old'] = pd.read_csv(f'{P.extras}/old_arms_rescored.csv')
        d['theirs_csv'] = pd.read_csv(glob.glob(P.eval_glob)[0], index_col=0)
        row = f'{ROOT}/repro/out/{cls}/phase3_gigapath_row.csv'
        d['giga'] = pd.read_csv(row).iloc[0] if os.path.exists(row) else None
    return d


EXPECTED_CHANGED = ('repro/COLAB_STEPS.md', 'repro/colab_gigapath.ipynb')   # edited in commit 197b23c (Colab notebook for both classes)


def verify_hashes(hash_file=HASH_BEFORE, expected_changed=()):
    """(total, identical, expected-changed files that differ, unexpected differences)."""
    import hashlib
    total = identical = 0
    changed, problems = [], []
    for line in open(hash_file).read().splitlines():
        digest, rel = line.split(maxsplit=1)
        total += 1
        h = hashlib.sha256()
        try:
            with open(f'{ROOT}/{rel}', 'rb') as fh:
                while blk := fh.read(1 << 20):
                    h.update(blk)
            ok = h.hexdigest() == digest
        except OSError:
            ok = False
        if ok:
            identical += 1
        elif rel in expected_changed:
            changed.append(rel)
        else:
            problems.append(rel)
    return total, identical, changed, problems


def versions():
    out = {'python': platform.python_version()}
    for label, mod in (('torch', 'torch'), ('torchvision', 'torchvision'), ('timm', 'timm'), ('numpy', 'numpy'), ('scikit-learn', 'sklearn'),
                       ('umap-learn', 'umap'), ('scipy', 'scipy'), ('pandas', 'pandas'), ('matplotlib', 'matplotlib'),
                       ('transformers', 'transformers'), ('huggingface_hub', 'huggingface_hub'), ('numba', 'numba'), ('pynndescent', 'pynndescent')):
        try:
            out[label] = importlib.import_module(mod).__version__
        except Exception:  # noqa: BLE001
            out[label] = 'not installed'
    return out


def extraction_minutes(paths):
    """Wall minutes per extraction from run_extractions*.log timestamps (read in order; a later 'start' replaces an earlier one)."""
    ev, rows = {}, {}
    for path in paths:
        for line in open(path):
            parts = line.split()
            if len(parts) < 3 or parts[1] not in ('start', 'done'):
                continue
            t, kind, rest = parts[0], parts[1], ' '.join(parts[2:])
            if kind == 'start':
                ev[rest] = t
            elif rest in ev:
                h0, m0, s0 = map(int, ev[rest].split(':'))
                h1, m1, s1 = map(int, t.split(':'))
                rows[rest] = ((h1 * 3600 + m1 * 60 + s1) - (h0 * 3600 + m0 * 60 + s0)) % 86400 / 60
    return rows


# --------------------------------------------------------------------------- report
def build_report(cls, fig_paths_aca, fig_paths_n, note_aca, note_n):
    A, A8, N = load(cls), load(cls, umap_dim=8), load('colon_n')
    PA, PN = A['P'], N['P']
    L = []

    def w(*parts, sep='\n'):
        L.append(sep.join(map(str, parts)))
    commit = open(f'{ROOT}/repro/UPSTREAM_COMMIT.txt').read().strip()
    w(f'# Reproduction of Batchkala, Li & Rittscher (LC25000-clean) on {cls}, next to colon_n')
    w('')
    w(f'Same upstream commit `{commit[:7]}`, same `venv_repro`, same patches as `repro/PATCHES.md` (none to upstream source), same criteria and thresholds as `repro/CRITERIA.md`.')
    w('colon_n columns are read from the existing colon_n outputs and never recomputed.')
    w('Prov-GigaPath is the 12th benchmark combo; its features come from a Colab T4 GPU run (provenance and caveat in sections 4 and 12).')
    w('')
    w('## 0. Setup')
    w('')
    w(*md(['', cls, 'colon_n'], [
        ['images', 5000, 5000],
        ['their final groups (`final_clusters.json`)', PA.k_final, PN.k_final],
        ['their saved initial clustering', f'UMAP-{PA.their_dim} + KMeans k=250', f'UMAP-{PN.their_dim} + KMeans k=250'],
        ['their sklearn (from the .pkl) / n_init', f"{A['p1']['their_sklearn_version']} / {A['p1']['their_n_init']}", f"{N['p1']['their_sklearn_version']} / {N['p1']['their_n_init']}"],
        ['my sklearn / n_init of my KMeans', f"{versions()['scikit-learn']} / {A['p1']['my_n_init']}", f"{versions()['scikit-learn']} / {N['p1']['my_n_init']}"],
    ]), sep='\n')
    w('')
    w(f'**The saved colon_aca clustering is UMAP-{PA.their_dim}, not UMAP-8.** The primary Phase 1 run for colon_aca therefore uses UMAP-{PA.their_dim}. A UMAP-8 run is reported separately and labelled "extra" for a like-for-like row against colon_n.')
    w('')

    # ---- C1
    def c1_rows(crit_list):
        names = list(crit_list[0]['clusters']['floors'])
        rows = [['my seed-42 ARI vs their .pkl', *[f4(c['clusters']['my_seed42_ARI_vs_their_pkl']) for c in crit_list]]]
        for nm in names:
            cells = []
            for c in crit_list:
                f = c['clusters']['floors'][nm]
                cells.append(f"{f['mean']:.4f} ± {f['sd']:.4f}, band [{f['lo']:.4f}, {f['hi']:.4f}]: {'PASS' if f['pass'] else 'FAIL'}")
            rows.append([nm, *cells])
        rows.append(['**C1**', *[('PASS' if c['clusters']['pass_all'] else 'FAIL') + ' (' + str(sum(f['pass'] for f in c['clusters']['floors'].values())) + '/3)' for c in crit_list]])
        return rows
    w('## 1. Criteria (thresholds unchanged from CRITERIA.md)')
    w('')
    w('### C1: initial clusters reproduced (my seed-42 ARI vs their .pkl inside my own seed-to-seed band, mean ± 2 sd)')
    w('')
    w(*md(['', f'{cls} primary (UMAP-{PA.umap_dim})', f'{cls} extra (UMAP-8)', f'colon_n (UMAP-{PN.umap_dim})'],
          c1_rows([A['crit'], A8['crit'], N['crit']])), sep='\n')
    w('')
    w('### C2: features reproduced (UNI resize_only benchmark row)')
    w('')
    rows = []
    for m in ('precision@1', 'precision@5', 'Adjusted Rand Index (ARI)', 'Fowlkes-Mallows Index'):
        a, n = A['crit']['features'][m], N['crit']['features'][m]
        rows.append([m, f4(a['theirs'], 5), f4(a['mine'], 5), f"{a['delta']:+.5f}", a['bar'], 'PASS' if a['pass'] else 'FAIL',
                     f"{n['delta']:+.5f}", 'PASS' if n['pass'] else 'FAIL'])
    w(*md(['metric', f'{cls} theirs', f'{cls} mine', f'{cls} delta', 'bar', f'{cls} result', 'colon_n delta', 'colon_n result'], rows), sep='\n')
    w('')

    # ---- Phase 1
    def p1col(d):
        p, an = d['p1'], d['anc']
        nf, uo, kk = p['noise_full_pipeline'], d['um'], p['noise_kmeans_only_fixed_umap42']
        sm, st = p['size_stats_mine'], p['size_stats_theirs']
        return [f4(p['my_vs_theirs']['ARI']), f4(p['my_vs_theirs']['AMI']), f"{p['my_vs_theirs']['n_identical_clusters']}/250",
                f4(p['my_vs_theirs']['frac_images_in_matched_cluster']), f"{p['anchors_equal_to_theirs']}/250",
                f"{an['anchor_equal_within_identical']}/{an['identical_clusters']}", f"{an['their_anchor_in_matched_my_cluster']}/250",
                f4(p['rerun_seed42_vs_mine']['ARI']),
                f"{sm['min']:.0f} / {sm['median']:.0f} / {sm['max']:.0f} / {sm['sd']:.2f}", f"{st['min']:.0f} / {st['median']:.0f} / {st['max']:.0f} / {st['sd']:.2f}",
                f"{p['sorted_size_L1']}",
                f"{nf['ARI_among_own_mean']:.4f} ± {nf['ARI_among_own_sd']:.4f}", f"{uo['ARI_among_own_mean']:.4f} ± {uo['ARI_among_own_sd']:.4f}",
                f"{kk['ARI_among_own_mean']:.4f} ± {kk['ARI_among_own_sd']:.4f}"]
    labels = ['ARI mine vs theirs', 'AMI mine vs theirs', 'clusters identical as image sets', 'images in Hungarian-matched cluster', 'anchors equal to theirs',
              'anchors equal within identical clusters', 'their anchor lies in my matched cluster', 'determinism: rerun seed 42 vs mine (ARI)',
              'my sizes min / median / max / sd', 'their sizes min / median / max / sd', 'sorted-size L1 distance',
              'noise floor combined (ARI among my 10 runs)', 'noise floor UMAP-only', 'noise floor KMeans-only']
    cols = [p1col(A), p1col(A8), p1col(N)]
    w('## 2. Phase 1: initial clustering (UMAP + KMeans k=250, seed 42, upstream code)')
    w('')
    w(*md(['', f'{cls} primary (UMAP-{PA.umap_dim})', f'{cls} extra (UMAP-8)', f'colon_n (UMAP-{PN.umap_dim})'],
          [[lab, *[c[i] for c in cols]] for i, lab in enumerate(labels)]), sep='\n')
    w('')

    # ---- Phase 2
    def p2col(d):
        p = d['p2']
        return [p['n_pre_merge'], f"{p['n_accepted']} / {p['n_rejected_pure']}", p['n_final'], p['n_merges_implied'], len(p['merged_groups']),
                p['rejected_images'], p['clusters_with_rejection'],
                f"{p['final_groups_equal_single_initial_cluster']} ({p['final_groups_equal_single_initial_cluster'] / p['n_final']:.3f})",
                f"{p['final_groups_equal_single_pre_merge_cluster']} ({p['final_groups_equal_single_pre_merge_cluster'] / p['n_final']:.3f})",
                len(p['split_pre_merge_clusters']), len(p['final_groups_not_exact_union']),
                p['rejected_pure_alone_final'], p['rejected_pure_merged_with_accepted'], json.dumps(p['merge_composition'])]
    l2 = ['pre-merge clusters', 'accepted / rejected-pure clusters', 'final groups', 'merge operations implied', 'final groups made of >1 pre-merge cluster',
          'rejected images', 'initial clusters with ≥1 rejected image', 'final groups equal to ONE initial cluster unchanged (fraction)',
          'final groups equal to ONE pre-merge cluster (rejections allowed)', 'pre-merge clusters split across final groups',
          'final groups that are not an exact union of pre-merge clusters', 'rejected-pure clusters that stay alone as a final group',
          'rejected-pure clusters merged into an accepted cluster', 'merge composition']
    c2, c2n = p2col(A), p2col(N)
    w('## 3. Phase 2: rebuild of their final groups from their manual-review records')
    w('')
    w(*md(['', cls, 'colon_n'], [[l, a, b] for l, a, b in zip(l2, c2, c2n)]), sep='\n')
    w('')

    # ---- Phase 3 benchmark
    cmpA, cmpN = A['cmp'], N['cmp']
    rows = []
    for key in list(A['theirs_csv'].index):
        combo = key.replace('#euclidean#NoReduction#kmeans', '')
        th = A['theirs_csv'].loc[key]
        if combo in cmpA.index:
            r = cmpA.loc[combo]
            cells = [f4(r['ARI_theirs']), f4(r['ARI_mine']), f"{r['ARI_delta']:+.4f}", f"{r['FMI_delta']:+.4f}", f"{r['P@1_delta']:+.4f}", f"{r['P@5_delta']:+.4f}",
                     'PASS' if r['C2_pass_all4'] else 'FAIL']
        elif combo == 'prov-gigapath#imagenet' and A['giga'] is not None:
            g = A['giga']
            cells = [f4(g['ARI_theirs']), f4(g['ARI_mine']), f"{g['ARI_delta']:+.4f}", f"{g['FMI_delta']:+.4f}", f"{g['P@1_delta']:+.4f}", f"{g['P@5_delta']:+.4f}",
                     ('PASS' if g['C2_pass_all4'] else 'FAIL') + ' (Colab GPU)']
        else:
            cells = [f4(float(th['Adjusted Rand Index (ARI)'])), '—', '—', '—', '—', '—', '—']
        if combo in cmpN.index:
            rn = cmpN.loc[combo]
            cells += [f"{rn['ARI_delta']:+.4f}", 'PASS' if rn['C2_pass_all4'] else 'FAIL']
        elif combo == 'prov-gigapath#imagenet' and N['giga'] is not None:
            g = N['giga']
            cells += [f"{g['ARI_delta']:+.4f}", ('PASS' if g['C2_pass_all4'] else 'FAIL') + ' (Colab GPU)']
        else:
            cells += ['—', '—']
        rows.append([combo, *cells])
    w(f'## 4. Phase 3: benchmark (`evaluate_clustering.py`, NoReduction, euclidean, KMeans, k = {PA.k_final} for {cls} / {PN.k_final} for colon_n)')
    w('')
    w(*md(['combo', f'{cls} their ARI', f'{cls} my ARI', f'{cls} ΔARI', 'ΔFMI', 'ΔP@1', 'ΔP@5', f'{cls} C2 (all 4 bars)', 'colon_n ΔARI', 'colon_n C2'], rows), sep='\n')
    w('')

    def summ(d):
        return [f"{int(((d['P@1_delta'].abs() <= 0.002) & (d['P@5_delta'].abs() <= 0.002)).sum())}/{len(d)}",
                f"{int(((d['ARI_delta'].abs() <= 0.005) & (d['FMI_delta'].abs() <= 0.005)).sum())}/{len(d)}",
                f"{d['ARI_delta'].abs().mean():.4f}", f"{d['ARI_delta'].abs().max():.4f}", f"{int(d['C2_pass_all4'].sum())}/{len(d)}"]

    def with_giga(cmp, g):
        keep = ['ARI_delta', 'FMI_delta', 'P@1_delta', 'P@5_delta', 'C2_pass_all4']
        return cmp[keep] if g is None else pd.concat([cmp[keep], pd.DataFrame([g[keep]])], ignore_index=True)
    s11 = [summ(cmpA), summ(cmpN)]
    s12 = [summ(with_giga(cmpA, A['giga'])), summ(with_giga(cmpN, N['giga']))]
    labs = ['P@1 and P@5 within 0.002', 'ARI and FMI within 0.005', 'mean abs ΔARI', 'max abs ΔARI', 'all four bars']
    w(*md(['summary', f'{cls}, 11 CPU combos', f'{cls}, 12 (+ prov-gigapath)', 'colon_n, 11 CPU combos', 'colon_n, 12 (+ prov-gigapath)'],
          [[l, s11[0][i], s12[0][i], s11[1][i], s12[1][i]] for i, l in enumerate(labs)]), sep='\n')
    w('')
    if A['giga'] is not None and N['giga'] is not None:
        ga, gn = A['giga'], N['giga']
        rows = []
        for sname, lab in (('P@1', 'precision@1'), ('P@5', 'precision@5'), ('ARI', 'ARI'), ('FMI', 'FMI')):
            rows.append([lab, ga[f'{sname}_bar'], f4(ga[f'{sname}_theirs'], 5), f4(ga[f'{sname}_mine'], 5), f"{ga[f'{sname}_delta']:+.5f}", 'PASS' if ga[f'{sname}_pass'] else 'FAIL',
                         f4(gn[f'{sname}_theirs'], 5), f4(gn[f'{sname}_mine'], 5), f"{gn[f'{sname}_delta']:+.5f}", 'PASS' if gn[f'{sname}_pass'] else 'FAIL'])
        w('### Prov-GigaPath row (Colab T4 GPU features)')
        w('')
        w(*md(['metric', 'bar', f'{cls} theirs', f'{cls} mine', f'{cls} delta', f'{cls} result', 'colon_n theirs', 'colon_n mine', 'colon_n delta', 'colon_n result'], rows), sep='\n')
        w('')
        w(f"KMeans seed context (seeds 0..9, n_init default, these features): {cls} ARI {ga['ARI_seeds_mean']:.4f} ± {ga['ARI_seeds_sd']:.4f}, their ARI z = {ga['z_theirs_ARI']:+.2f} "
          f"({'inside' if ga['theirs_ARI_within_2sd'] else 'outside'} 2 sd); colon_n ARI {gn['ARI_seeds_mean']:.4f} ± {gn['ARI_seeds_sd']:.4f}, z = {gn['z_theirs_ARI']:+.2f} ({'inside' if gn['theirs_ARI_within_2sd'] else 'outside'} 2 sd).")
        w('')
        w('**Caveat:** these features were extracted on a Colab T4 GPU (torch 2.14.1+cu130, timm 1.0.31, fp32, batch 32, upstream 912f342), unlike the other 11 combos (CPU, local `venv_repro`: torch 2.14.1+cpu, timm 1.0.30). The last float digits can differ. File hashes: `repro/gigapath_colab_sha256.txt`.')
        w('')

    # ---- seeds UNI
    w(f'## 5. KMeans seed / n_init spread (UNI resize_only, NoReduction, seeds 0..9, scored vs their final groups)')
    w('')
    rows = []
    for n_init in ('auto', '10'):
        for m in ('ARI', 'FMI'):
            a, n = A['seeds'][n_init][m], N['seeds'][n_init][m]
            rows.append([f'n_init={n_init}', m, f"{a['mean']:.4f} ± {a['sd']:.4f} [{a['min']:.4f}, {a['max']:.4f}]", f4(a['theirs'], 5), f"{a['z_theirs']:+.2f}",
                         f"{a['n_runs_ge_theirs']}/10", f"{n['mean']:.4f} ± {n['sd']:.4f}", f4(n['theirs'], 5), f"{n['z_theirs']:+.2f}", f"{n['n_runs_ge_theirs']}/10"])
    w(*md(['KMeans', 'metric', f'{cls} mean ± sd [min, max]', f'{cls} theirs', f'{cls} z', f'{cls} runs ≥ theirs', 'colon_n mean ± sd', 'colon_n theirs', 'colon_n z', 'colon_n runs ≥ theirs'], rows), sep='\n')
    w('')
    na_a, na_n = A['noise_all'], N['noise_all']
    rows = []
    for combo in na_a.index:
        r = na_a.loc[combo]
        n = na_n.loc[combo] if combo in na_n.index else None
        rows.append([combo, f4(r['ARI_theirs']), f"{r['ARI_seeds_mean']:.4f} ± {r['ARI_seeds_sd']:.4f}", f"{r['z_theirs']:+.2f}", 'yes' if r['theirs_within_2sd'] else 'no',
                     '—' if n is None else f"{n['z_theirs']:+.2f}", '—' if n is None else ('yes' if n['theirs_within_2sd'] else 'no')])
    if A['giga'] is not None:
        ga, gn = A['giga'], N['giga']
        rows.append([COMBO_GIGA, f4(ga['ARI_theirs']), f"{ga['ARI_seeds_mean']:.4f} ± {ga['ARI_seeds_sd']:.4f}", f"{ga['z_theirs_ARI']:+.2f}", 'yes' if ga['theirs_ARI_within_2sd'] else 'no',
                     '—' if gn is None else f"{gn['z_theirs_ARI']:+.2f}", '—' if gn is None else ('yes' if gn['theirs_ARI_within_2sd'] else 'no')])
    w('Per-combo seed noise (n_init default, seeds 0..9) in context of their published ARI:')
    w('')
    w(*md(['combo', f'{cls} their ARI', f'{cls} my seeds mean ± sd', f'{cls} z', f'{cls} within 2 sd', 'colon_n z', 'colon_n within 2 sd'], rows), sep='\n')
    w('')

    # ---- k comparison
    w(f'## 6. k = {PA.k_final} (their group count) vs k = 250 (UNI resize_only, scored vs their final groups)')
    w('')
    rows = []
    for d, lab in ((A, cls), (N, 'colon_n')):
        kf = str(d['P'].k_final)
        for k in (kf, '250'):
            e = d['k'][k]
            s = e['seed42_upstream_metrics']
            rows.append([lab, f'k={k}', f4(s['Adjusted Rand Index (ARI)']), f4(s['Fowlkes-Mallows Index']), f4(s['Normalized Mutual Information (NMI)']),
                         f"{e['seeds0_9_mean']['Adjusted Rand Index (ARI)']:.4f} ± {e['seeds0_9_sd']['Adjusted Rand Index (ARI)']:.4f}"])
    w(*md(['class', 'k', 'ARI (seed 42)', 'FMI', 'NMI', 'ARI seeds 0..9 mean ± sd'], rows), sep='\n')
    w('')

    # ---- old arms
    w('## 7. My four old arms re-scored with their metric functions')
    w('')
    rows = []
    for arm in ('uni1_full', 'uni1_pca128', 'resnet50_full', 'resnet50_pca128'):
        cells = []
        for d in (A, N):
            o = d['old']
            up = o[(o['arm'] == arm) & o['setting'].str.startswith('upstream_pipeline')].iloc[0]
            ol = o[(o['arm'] == arm) & (o['setting'] == 'old_k250_labels_rescored')].iloc[0]
            cells += [f4(up['Adjusted Rand Index (ARI)']), f4(ol['Adjusted Rand Index (ARI)'])]
        rows.append([arm, *cells])
    w(*md(['arm', f'{cls}: upstream pipeline k={PA.k_final}, ARI', f'{cls}: my old k=250 labels, ARI', f'colon_n: upstream pipeline k={PN.k_final}, ARI', 'colon_n: my old k=250 labels, ARI'], rows), sep='\n')
    w('')

    # ---- figures
    w('## 8. Cluster-size distributions')
    w('')
    w(f'Four distributions per class: their final groups, their initial 250, my initial 250, my benchmark KMeans run (UNI resize_only, k = group count, seed 42). Figures: `repro/figs/{cls}/` and `repro/figs/colon_n/`.')
    w(f'The colon_n "my benchmark KMeans run" sizes were not stored; they were re-derived in memory from the existing feature vectors ({note_n}). colon_aca: {note_aca}.')
    w('')
    names = ['sizes_their_final', 'sizes_their_initial250', 'sizes_my_initial250', 'sizes_my_benchmark_kmeans']
    for nm in names:
        w(f'![{cls} {nm}](figs/{cls}/{nm}.png)')
    w('')
    for nm in names:
        w(f'![colon_n {nm}](figs/colon_n/{nm}.png)')
    w('')

    # ---- runtimes
    rt = pd.read_csv(f'{PA.out}/step_runtimes.csv')
    ext_a = extraction_minutes(sorted(glob.glob(f'{ROOT}/repro/logs/{cls}/run_extractions*.log')))
    ext_n = extraction_minutes([f'{ROOT}/repro/logs/run_extractions.log'])
    w('## 9. Runtimes (wall clock on this machine, 20 cores, CPU only)')
    w('')
    rows = [[r.phase, r.step, r.seconds, f'{r.seconds / 60:.1f}', r.exit_code] for r in rt.itertuples()]
    rows.append(['', '**total of the steps above**', int(rt['seconds'].sum()), f"{rt['seconds'].sum() / 60:.1f}", ''])
    w(*md(['phase', f'step ({cls})', 'seconds', 'minutes', 'exit'], rows), sep='\n')
    w('')
    w('Feature extraction per combo (from the `run_extractions` log timestamps; `extract_*` rows above are the two Phase 1 extractions run directly):')
    w('')
    combos = sorted(set(ext_a) | set(ext_n))
    w(*md(['combo', f'{cls} minutes', 'colon_n minutes'],
          [[c, f'{ext_a[c]:.1f}' if c in ext_a else '', f'{ext_n[c]:.1f}' if c in ext_n else ''] for c in combos]), sep='\n')
    w('')
    note = f'{ROOT}/repro/out/{cls}/NOTE_resume.txt'
    if os.path.exists(note):
        w('Interruptions during the UNI#imagenet extraction (it saves features only at the end, so each kill lost the whole run):')
        w('')
        for line in open(note).read().splitlines():
            line = line.strip().lstrip('# ').strip()
            if line:
                w(f'- {line}')
        w('')
        w('The UNI#imagenet time in the tables is from the final clean run (a user-level systemd service). `repro/logs/colon_aca/run_extractions_part1.log` and `run_extractions_resume1_killed.log` keep the logs of the killed attempts.')
        w('')
    w('colon_n timings exist only for extractions run through `run_extractions.sh`; its UNI#imagenet figure spans an overnight pause and is not a compute time. Other colon_n steps were not timed.')
    w('')

    # ---- versions
    v = versions()
    w('## 10. Versions')
    w('')
    w(*md(['package', 'version'], [[k, x] for k, x in v.items()] + [['their .pkl sklearn', A['p1']['their_sklearn_version']], ['upstream commit', commit], ['Prov-GigaPath (Colab T4)', 'torch 2.14.1+cu130, timm 1.0.31']]), sep='\n')
    w('')

    # ---- hash check
    total, identical, changed, problems = verify_hashes(HASH_BEFORE, EXPECTED_CHANGED)
    t2, i2, _, p2_ = verify_hashes(f'{ROOT}/markers/colon_n_hashes_before.txt')
    bad = len(problems) + len(p2_)
    w('## 11. colon_n integrity (hash report)')
    w('')
    w(f"- `markers/colon_n_repro_hashes_before.txt`: {identical}/{total} files identical; {len(changed)} expected changes ({', '.join(f'`{c}`' for c in changed) or 'none'}, edited in commit 197b23c); unexpected differences or missing files: {'none' if not problems else problems[:10]}.")
    w(f"- `markers/colon_n_hashes_before.txt` (the colon_n pipeline: `embeddings/`, `results/`, `data/final`, markers): {i2}/{t2} files identical; unexpected differences: {'none' if not p2_ else p2_[:10]}.")
    w('- Covered: `repro/out/**`, `repro/scratch_eval*/**`, `repro/logs/*` (top level), the `repro/*.md|txt|ipynb` docs, `install.sh`, upstream `feature_vectors/colon_n/**`, `eval_results/**`, `annotations/**`, and the 5000 colon_n originals.')
    w('- colon_n scripts re-run with `REPRO_OUT_ROOT` against a scratch copy of `repro/out` reproduced `phase2_metrics.json`, `pre_merge_clusters.json`, `anchor_extra.json`, `criteria_check.json` and `phase3_compare.csv` byte for byte after the `--cls` change.')
    w('- New, not checked:')
    for p in fig_paths_n:
        w(f'  - `{p}`')
    for p in ('repro/out/colon_n/phase3_gigapath_row.csv', 'repro/out/colon_n/phase3_gigapath_eval.csv', 'repro/out/colon_n/phase3_gigapath_eval.json',
              'repro/out/colon_n/GIGAPATH_NOTE.md', 'repro/gigapath_colab_sha256.txt',
              'third_party/LC25000-clean/feature_vectors/colon_n/prov-gigapath/imagenet/{features.npy,ids.npy,ids_2_img_paths.json} (gitignored)'):
        w(f'  - `{p}`')
    w('- The Prov-GigaPath evaluation ran in `repro/scratch_eval/gigapath/<cls>/` (new, gitignored); upstream `eval_results/` and the existing `repro/scratch_eval/**` were not written.')
    w('')
    w('## 12. What differs from colon_n (read before comparing numbers)')
    w('')
    w(f'- Their colon_aca initial clustering is UMAP-{PA.their_dim}; colon_n is UMAP-{PN.their_dim}. The extra UMAP-8 columns give a like-for-like row.')
    w(f"- Their colon_aca review is much lighter: {A['p2']['rejected_images']} rejected images in {A['p2']['clusters_with_rejection']} clusters, against {N['p2']['rejected_images']} in {N['p2']['clusters_with_rejection']} for colon_n.")
    w('- Prov-GigaPath features (both classes) come from a Colab T4 GPU run, the other 11 combos from local CPU runs; see the caveat under the Prov-GigaPath table in section 4.')
    w('')
    out = f'{ROOT}/repro/REPORT_{cls}.md'
    open(out, 'w').write('\n'.join(L))
    print(f'[report] wrote {out} | hash check: repro {identical}/{total} ({len(changed)} expected changes), pipeline {i2}/{t2}')
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--figs', choices=('colon_n', 'colon_aca'), help='figures only')
    ap.add_argument('--cls', choices=('colon_aca',), help='figures for colon_aca and colon_n, plus the report')
    a = ap.parse_args()
    if a.figs:
        make_figures(a.figs)
        return 0
    if a.cls:
        figs_a, note_a = make_figures(a.cls)
        figs_n, note_n = make_figures('colon_n')
        return 1 if build_report(a.cls, figs_a, figs_n, note_a, note_n) else 0
    ap.error('give --figs or --cls')


if __name__ == '__main__':
    sys.exit(main())
