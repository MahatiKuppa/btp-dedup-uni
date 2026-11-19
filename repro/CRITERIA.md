# Reproduction criteria

> **Written AFTER seeing the Phase 1 results and the first benchmark row (UNI resize_only).**
> The cluster criterion is a post-hoc rule built from the Phase 1 noise floors. The feature thresholds
> (0.002 for P@k, ~0.005 for ARI/FMI) were the user's informal bars set before seeing results; they are written here unchanged and **not loosened**.
> Everything below is evaluated by `repro/criteria_check.py` (output `repro/out/criteria_check.json`) and `repro/stage_c_kmeans_seeds_uni.py`.

## C1. Clusters reproduced
My seed-42 initial clustering (UMAP-8 + KMeans k=250, upstream code) has ARI vs their `.pkl` labels inside my own seed-to-seed ARI range,
for each of three noise floors, as mean ± 2 sd of the pairwise ARI among my 10 runs (seeds 0..9, sd = population sd of the 45 pairwise ARIs; pairs are not independent, so the band is only a rough guide):
combined (UMAP+KMeans seeds), UMAP-only (KMeans seed 42), KMeans-only (UMAP seed 42).

## C2. Features reproduced (UNI resize_only row of the benchmark, upstream `evaluate_clustering.py`, k=257)
- |delta| of P@1 and P@5 <= 0.002
- |delta| of ARI and FMI <= ~0.005 (taken as 0.005)

## Results against the criteria

### C1 (my seed-42 ARI vs their pkl = 0.9308)
| noise floor | mean ± sd (ARI among my runs) | band mean ± 2 sd | result |
|---|---|---|---|
| combined | 0.9215 ± 0.0112 | [0.8991, 0.9438] | PASS |
| UMAP-only | 0.9295 ± 0.0081 | [0.9133, 0.9458] | PASS |
| KMeans-only | 0.9412 ± 0.0063 | [0.9286, 0.9538] | PASS (close to the lower edge: 0.9308 vs 0.9286) |

**C1: PASS (3/3).** Initial clustering is reproduced up to run-to-run noise. It is not bit-identical (183/250 clusters identical as image sets, 25/250 anchors equal).

### C2
| metric | theirs | mine | delta | bar | result |
|---|---|---|---|---|---|
| precision@1 | 0.99720 | 0.99720 | 0.00000 | 0.002 | PASS |
| precision@5 | 0.98516 | 0.98508 | -0.00008 | 0.002 | PASS |
| ARI | 0.63765 | 0.64344 | +0.00579 | 0.005 | **FAIL (marginal)** |
| FMI | 0.64027 | 0.64638 | +0.00611 | 0.005 | **FAIL (marginal)** |

**C2: P@1/P@5 PASS; ARI and FMI fail the bar by ~0.001.** The bar is not loosened. The ARI/FMI gap is then investigated below as a separate question.

### Does KMeans seed / n_init variation at k=257 explain the ARI/FMI gap? (UNI resize_only, NoReduction, euclidean, scored vs their 257 final groups)
Seeds 0..9, sd with ddof=1. Seed 42 / n_init='auto' reproduces the benchmark number exactly (0.64344), confirming the setup.

| KMeans setting | ARI mean ± sd | ARI range | FMI mean ± sd | their ARI 0.63765: z | runs with ARI >= theirs |
|---|---|---|---|---|---|
| n_init='auto' (=1 in sklearn>=1.4; what upstream code does) | 0.6628 ± 0.0110 | [0.6430, 0.6759] | 0.6650 ± 0.0108 | -2.29 | 10/10 |
| n_init=10 | 0.6781 ± 0.0156 | [0.6519, 0.6944] | 0.6802 ± 0.0154 | -2.59 | 10/10 |

Reading (stated plainly):
- **Partly explained.** Seed-to-seed sd of ARI (0.011) is about twice the +0.0058 gap, so the gap between my seed-42 value and theirs is not distinguishable from KMeans seed noise. My seed 42 is itself one of the lowest draws (second lowest of 11; seed 3 = 0.6430 is lowest).
- **Not fully explained.** Their 0.63765 is below all 10 of my 'auto' draws (z = -2.3) and all 10 n_init=10 draws (z = -2.6). With only 10 seeds the tail estimate is crude, but a lower-than-typical draw alone is not a comfortable explanation. A small systematic shift (their GPU-extracted features, sklearn 1.4.2 vs 1.9.1 KMeans internals) is not excluded. P@1/P@5 agreement says the features are near-identical; the sklearn-version effect is untested.
- n_init=10 raises ARI by ~0.015 on average, so upstream's unset n_init (sklearn default) matters more than the ARI/FMI bar.

## How these criteria will be used in REPORT.md
Phase 3 rows (11 now, 12 after Colab prov-gigapath) are reported with deltas against the C2 bars per combo, **and** with the UNI KMeans-seed sd (about 0.011 ARI) as context, without relabelling any FAIL.

## Phase 3 results against C2 (11 of 12 combos; prov-gigapath pending the Colab features)
Source: `repro/out/phase3_compare.csv`, `repro/out/phase3_kmeans_seed_noise_all.csv` (written after seeing these numbers; bars unchanged).
- **P@1 and P@5: PASS in 11/11 combos** (max |delta| 0.0008). Features are reproduced.
- **ARI and FMI within 0.005: PASS in 1/11 combos** (owkin-phikon#imagenet). The other 10 fail, with |ARI delta| from 0.0057 up to 0.0358 (ResNet50-CLAM). Mean |ARI delta| = 0.0135.
- The 0.005 bar is smaller than KMeans seed noise for most combos (sd of ARI over seeds 0..9: 0.002 to 0.016). Context, not a re-labelling: their ARI lies within 2 sd of my 10-seed distribution in 8/11 combos (mean z = +0.19, so no systematic sign); outside 2 sd: UNI#resize_only (z=-2.29), resnet18#resize_only (z=+2.73), resnet50-clam#imagenet (z=+2.36).
- Old arms (`repro/out/phase3_extras/`): my old UNI-full arm gives ARI 0.439 under their metric/pipeline, close to UNI#imagenet (0.450 theirs / 0.465 mine), not UNI#resize_only (0.638). The old pipeline used timm's default `resolve_data_config` transform (ImageNet normalisation) plus L2-normalised rows.
