# colon_aca vs colon_n: dedup pipeline summary

Same pipeline, same settings for both classes (UNI v1, ResNet50, KMeans k=250 random_state=42 n_init=10, PCA-128).
The colon_n columns are read from the existing results, not recomputed.

## Data and published groups

| | colon_aca | colon_n |
|---|---|---|
| images | 5000 | 5000 |
| published groups | 249 | 257 |
| group size min / max / mean | 6 / 34 / 20.08 | 1 / 53 / 19.46 |
| singleton groups | 0 | 6 |
| files with ambiguous their_filename (byte-identical duplicates) | 489 | 454 |
| representatives in data/final/colon_aca | 249 | 257 |

KMeans uses k=250 for both classes; colon_aca has 249 published groups and colon_n has 257, so ARI is scored against those.
Duplicate handling is identical to colon_n: byte-identical files are paired by sorted position within the hash, and stage 4 stops unless every duplicate set lies in one group_id.

## colon_aca: KMeans arms (k=250)

| arm | size_min | size_max | size_mean | size_std | separation_ratio | ari |
|---|---|---|---|---|---|---|
| uni1_full | 5 | 45 | 20.00 | 6.01 | 6.576 | 0.8571 |
| uni1_pca128 | 6 | 51 | 20.00 | 6.19 | 9.168 | 0.8172 |
| resnet50_full | 4 | 62 | 20.00 | 8.67 | 4.392 | 0.6711 |
| resnet50_pca128 | 4 | 66 | 20.00 | 9.00 | 5.794 | 0.6741 |

## colon_n: KMeans arms (k=250, existing results)

| arm | size_min | size_max | size_mean | size_std | separation_ratio | ari |
|---|---|---|---|---|---|---|
| uni1_full | 3 | 70 | 20.00 | 10.17 | 5.745 | 0.4653 |
| uni1_pca128 | 3 | 71 | 20.00 | 10.23 | 7.157 | 0.4310 |
| resnet50_full | 3 | 69 | 20.00 | 10.10 | 4.337 | 0.4304 |
| resnet50_pca128 | 5 | 79 | 20.00 | 9.44 | 5.102 | 0.4523 |

Cluster size columns are over the 250 KMeans clusters. ARI is against the published groups.

## Stage runtimes (wall clock, colon_aca)

| stage | seconds | minutes | exit |
|---|---|---|---|
| stage_2_embed_uni | 197 | 3.3 | 1 |
| stage_2_embed_uni | 2255 | 37.6 | 0 |
| stage_2_embed_resnet | 208 | 3.5 | 0 |
| stage_3_kmeans_uni | 48 | 0.8 | 0 |
| stage_3_kmeans_uni_pca128 | 26 | 0.4 | 0 |
| stage_3_kmeans_resnet | 77 | 1.3 | 0 |
| stage_3_kmeans_resnet_pca128 | 26 | 0.4 | 0 |
| stage_4_match_groups | 3 | 0.1 | 0 |
| stage_5_compare | 1 | 0.0 | 0 |
| stage_6_final_dedup | 39 | 0.7 | 0 |
| total | 2880 | 48.0 | |

The first stage_2_embed_uni row (exit 1) is the first attempt: it downloaded the HF dataset (~1.9 GB), then stopped on the label check because the dataset's name is 'adenocarcinomas', not 'adenocarcinoma'. The mapping was fixed and the rerun reused the cached dataset, so the later stage_2_embed_uni row has no download time; the download is that first row's time.
Both stage 2 rows are kept. The total includes the failed attempt.

## Package versions

- python: 3.12.3
- torch: 2.14.0+cpu
- torchvision: 0.29.0+cpu
- timm: 0.9.8
- numpy: 2.5.3
- scikit-learn: 1.9.1
- umap-learn: 0.5.12
- scipy: 1.18.1
- pandas: 3.0.6
- matplotlib: 3.11.2
- opencv: not installed
- datasets: 3.6.0
- huggingface_hub: 0.36.2
- sharpness backend (stage 6): pil+numpy
- stage 2 UNI marker: timm 0.9.8, torch 2.14.0+cpu
- stage 2 ResNet50 marker: torchvision 0.29.0+cpu, torch 2.14.0+cpu
- stage 3 marker: sklearn 1.9.1, umap 0.5.12

## colon_n integrity check

- stage 5 `--check` on colon_n (recompute vs results/ground_truth_comparison.csv, no writes): PASS - [check] OK: identical to the stored values
- sha256 of colon_n files vs markers/colon_n_hashes_before.txt: PASS - 305/305 files identical (38 under embeddings/ and results/)
