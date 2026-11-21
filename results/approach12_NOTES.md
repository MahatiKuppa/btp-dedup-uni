# Approaches 1 and 2 (Colab, Lab histograms) scored against the published LC25000-clean groups

Script: `scripts/approach12_score.py` (steps map, stage3, sub, replay, score). Inputs: `data/approach12_drive/` (gitignored), copied from the Colab Drive:
the two notebooks, `approach1/logs/`, `approach2/logs/`, and the re-saved images `colon_raw_all.zip` (both zips have identical content: names, sizes and CRCs of all 10000 entries).
Approach 2 ran on colon_n only, so it has no colon_aca rows.

## Environment

- KMeans-dependent steps (stage3, sub, replay): sklearn 1.7.2, opencv 5.1.0, numpy 2.5.5, pandas 3.1.1, scipy 1.18.1, python 3.12.3.
- ARI/FMI scoring and the separation ratio: sklearn 1.9.1, numpy 2.5.5, pandas 3.1.1 (same sklearn as the approach 3 scores in stage 5). The map step ran in the 1.9.1 venv (opencv 5.1.0).
- Colab's own versions are not known: the notebooks have no saved pip output. Colab ran an sklearn that reproduces its KMeans (see next section); its opencv is unknown.

## Colab's KMeans is reproduced by sklearn 1.6.1 / 1.7.2, not by 1.8.0 / 1.9.1

On approach 2's saved `colon_n_features.npy` (bit-identical to the features I recompute from the re-saved images), `KMeans(n_clusters=250, random_state=42, n_init=10)` was fitted in a fresh venv per version and compared with the saved `colon_n_labels_baseline.npy`:

| sklearn | identical to saved labels | labels differing | ARI vs saved |
|---|---|---|---|
| 1.9.1 | no | 4717 | 0.8238 |
| 1.8.0 | no | 4717 | 0.8238 |
| 1.7.2 | yes | 0 | 1.0000 |
| 1.6.1 | yes | 0 | 1.0000 |

Versions 1.9.1 and 1.8.0 give a different partition (ARI 0.824 against the saved labels), so every KMeans step here ran in 1.7.2. Tried: 1.8.0, 1.7.2, 1.6.1 (3 versions, all installed cleanly, about 2 minutes in total); the 1.9.1 result comes from the main venv. `scripts/approach12_sklearn_check.py` is the check. The scratch venvs were deleted except the 1.7.2 one that was extended with opencv, pandas and pillow.

## Checks and outcomes

**1. File mapping (re-saved Colab file `i` = original `data/raw/<cls>/<cls>_{i:04d}.jpeg`): PASS for both classes.**
- colon_n: 5000 pairs, mean abs pixel difference mean 0.095, p99 0.216, max 0.260 (largest single-pixel difference 37). On 32x32 thumbnails the worst matched pair is at 0.09; the closest other original (byte-identical copies of the image excluded) is at 5.14. Own original is the nearest match for all 5000: True.
- colon_aca: 5000 pairs, mean abs pixel difference mean 0.098, p99 0.377, max 0.594 (largest single-pixel difference 38). On 32x32 thumbnails the worst matched pair is at 0.52; the closest other original (byte-identical copies of the image excluded) is at 3.81. Own original is the nearest match for all 5000: True.

**2. Stage 3 (Lab histograms + KMeans k=250), recomputed from the re-saved images with sklearn 1.7.2: PASS for both classes.**
- colon_n features vs saved `colon_n_features.npy`: exactly equal = True (max abs diff 0.0).
- colon_n recomputed labels vs saved `labels_baseline.npy`: identical = True (labels differing 0).
- colon_n: `n_members` of all 250 clusters equal `clusters.csv`: True. Representatives (sharpest member by Laplacian variance): 244/250 strict, 250/250 within 1e-12; strict mismatches 6 (max relative gap 2.4e-16).
- colon_aca: `n_members` of all 250 clusters equal `clusters.csv`: True. Representatives (sharpest member by Laplacian variance): 245/250 strict, 250/250 within 1e-12; strict mismatches 5 (max relative gap 1.4e-16).

**Representative rule.** The logged representative is accepted if it is a member of the cluster and its sharpness equals the cluster maximum within 1e-12 relative. All logged representatives satisfy this. The strict check (my argmax equals the logged pick) fails for the clusters above, always on two members whose sharpness is equal to float rounding, so which one `argmax` returns can depend on the last bits (possibly the opencv version: mine is 5.1.0, Colab's is unknown).
- colon_n: 9 strict mismatches (stage 3 and sub-KMeans together): 0 of the tied pairs are byte-identical duplicates in the original release (sha256 of the `data/raw` originals; none is flagged `their_filename_ambiguous` in the stage 4 mapping either); 9 of them lie in the same published group.
- colon_aca: 5 strict mismatches (stage 3 and sub-KMeans together): 0 of the tied pairs are byte-identical duplicates in the original release (sha256 of the `data/raw` originals; none is flagged `their_filename_ambiguous` in the stage 4 mapping either); 5 of them lie in the same published group.

**3. Approach 1 stage 4 (sub-KMeans of oversized clusters, n_members > 30), recomputed: PASS for both classes.**
- colon_n: 25 oversized clusters = the ids in `splits.csv`: True; sub-cluster count per cluster equal: True; (cluster, sub) keys equal: True; sizes equal: True; pieces 279 (expected 279); representatives: 51/54 strict, 54/54 within 1e-12; strict mismatches 3 (max relative gap 0.0e+00).
- colon_aca: 23 oversized clusters = the ids in `splits.csv`: True; sub-cluster count per cluster equal: True; (cluster, sub) keys equal: True; sizes equal: True; pieces 276 (expected 276); representatives: 49/49 strict, 49/49 within 1e-12; strict mismatches 0 (max relative gap 0.0e+00).

**4. Approach 1 final partition (logged merges joined with union-find over pieces): PASS for both classes.**
- colon_n: 36 logged merges (6 where a kept image is itself removed later, so union-find is needed); 243 groups = 243 rows of `final_manifest.csv`; surviving representatives (all pieces minus the removed names) equal `final_manifest.csv` as a set: True; unknown names in merges.csv: 0.
- colon_aca: 30 logged merges (2 where a kept image is itself removed later, so union-find is needed); 246 groups = 246 rows of `final_manifest.csv`; surviving representatives (all pieces minus the removed names) equal `final_manifest.csv` as a set: True; unknown names in merges.csv: 0.
- colon_n: the largest group of the rebuilt partition has 169 images (next largest 57); it joins 11 pieces that cover 11 different published groups.
- colon_aca: the largest group of the rebuilt partition has 84 images (next largest 52); it joins 5 pieces that cover 6 different published groups.

**5. Approach 1 stage 5 (ORB merge pass) replayed as a check only, from the logged representative file of every piece (not the re-picked one): the scored partition always uses the logged merges.**
- colon_n: 279 representative files, 1881 candidate pairs; replayed merges 36 vs logged 36: identical sequence (order, kept, removed, ORB score) = True.
- colon_aca: 276 representative files, 1876 candidate pairs; replayed merges 30 vs logged 30: identical sequence (order, kept, removed, ORB score) = True.

**6. Approach 2 saved labels.**
- `colon_n_labels_final.npy`: int64 (5000,), values 0..249 contiguous, 250 distinct (40,128 bytes = 5000 x 8 + 128 header; the notebook builds it from Python ints, the other label files are int32).
- Separation ratio recomputed with the notebook function on the saved arrays: baseline with the saved KMeans centres 76.44684600830078 vs logged 76.44684600830078 (equal: True); baseline with group-mean centres differs from logged by 0.00e+00; final 64.5816650390625 vs logged 64.5816650390625 (equal: True).
- The notebook's own result: separation ratio fell from 76.45 (baseline) to 64.58 (final), which is the opposite of its 'want up' criterion.

**7. Scoring path self-check (approach 3, UNI labels from stage 3 through the same code):**
- colon_n: ARI 0.46528819152130463 / FMI 0.4690940915380049 = stored stage 5 values: True.
- colon_aca: ARI 0.85706999769553 / FMI 0.8577525986489428 = stored stage 5 values: True.

## What was recomputed and what was read

- Read from file: `colon_n` `lab_kmeans250` (= `labels_baseline.npy`, what ran), `a2_final` (`labels_final.npy`), `a2_refined` (`labels_refined.npy`); the logs `clusters/splits/merges/final_manifest.csv`; the approach 2 metrics, features and centres used for the check in point 6.
- Recomputed (sklearn 1.7.2, opencv above): Lab features of both classes, the stage-3 KMeans of both classes (colon_n identical to the saved labels, used as a check only; colon_aca used for scoring), the sub-KMeans of the oversized clusters, the ORB replay (check only).
- Rebuilt from logs: `a1_final` and `a1_after_split` (stage-3 clusters, oversized ones replaced by their recomputed pieces, then the logged merges joined by union-find).
- Approach 1 has no saved label array and no separation ratio in its notebook; approach 2 has no colon_aca run.

## Results (ARI / FMI against the published groups; colon_n 257 groups, colon_aca 249)

Rows marked `extra` were added on request and are not part of the main comparison. `sep_ratio` is approach 2's function (inter / intra with group-mean centres) applied to every set; notebook 1 defines no ratio, so for `a1_*` rows it is an extra measurement, not a notebook value.

| class | label set | kind | clusters | size min / median / max | ARI | FMI | sep. ratio |
|---|---|---|---|---|---|---|---|
| colon_n | lab_kmeans250 | main | 250 | 4 / 19 / 67 | 0.7322 | 0.7345 | 76.45 |
| colon_n | a1_final | main | 243 | 6 / 19 / 169 | 0.6914 | 0.6981 | 33.42 |
| colon_n | a2_final | main | 250 | 10 / 19 / 65 | 0.7562 | 0.7573 | 64.58 |
| colon_n | a1_after_split | extra | 279 | 4 / 18 / 36 | 0.7680 | 0.7698 | 80.33 |
| colon_n | a2_refined | extra | 256 | 8 / 19 / 65 | 0.7577 | 0.7587 | 66.34 |
| colon_aca | lab_kmeans250 | main | 250 | 2 / 19 / 80 | 0.7857 | 0.7882 | 120.70 |
| colon_aca | a1_final | main | 246 | 2 / 19 / 84 | 0.8083 | 0.8104 | 69.05 |
| colon_aca | a1_after_split | extra | 276 | 2 / 18 / 36 | 0.8207 | 0.8217 | 133.65 |

Approach 3 (UNI/ResNet KMeans k=250, stage 5, read from the existing csv files, not recomputed) for reference:

| class | arm | ARI | FMI |
|---|---|---|---|
| colon_n | kmeans_uni | 0.4653 | 0.4691 |
| colon_n | kmeans_uni_pca128 | 0.4310 | 0.4349 |
| colon_n | kmeans_resnet | 0.4304 | 0.4342 |
| colon_n | kmeans_resnet_pca128 | 0.4523 | 0.4557 |
| colon_aca | kmeans_uni | 0.8571 | 0.8578 |
| colon_aca | kmeans_uni_pca128 | 0.8172 | 0.8181 |
| colon_aca | kmeans_resnet | 0.6711 | 0.6738 |
| colon_aca | kmeans_resnet_pca128 | 0.6741 | 0.6770 |

Files: `results/approach12_vs_ground_truth.csv`, `results/approach12_mapping_stats.csv` (per file pair), `results/approach12_checks.json` (all check details), `results/approach12_labels/*.npy` (every scored label array, row order = `embeddings/**/my_filename_to_group_id.csv`).
