# Discussion tests: rotation sensitivity (Test A) and ORB merges (Test B)

Script: `scripts/discussion_tests.py` (steps lab, pk, uni, orb, report, plot). Lab and ORB work ran in `venv_sk_1.7.2` (opencv 5.1.0, sklearn 1.7.2), UNI in `venv` (timm 0.9.8, torch 2.14.0+cpu, CPU, fp32), the figure in `venv_repro`.
Data: the original files in `data/raw/<cls>`. Published groups from `embeddings/**/my_filename_to_group_id.csv`. Approach 1 logs and partitions from `data/approach12_drive/`.

## Test A1: P@1 / P@5 in Lab-histogram space

Upstream definition (`precision_at_1`, `precision_at_k`: Euclidean on the features as given, the image itself excluded, mean of the 0/1 same-published-group flag over all images and neighbours), copied verbatim into the script because the upstream module imports `umap`. The copy reproduces my stored UNI values from the stored features, per class: colon_n imagenet: P@1 0.99420, P@5 0.94576 (stored 0.99420 / 0.94576); colon_n resize_only: P@1 0.99720, P@5 0.98508 (stored 0.99720 / 0.98508); colon_aca imagenet: P@1 0.99920, P@5 0.99624 (stored 0.99920 / 0.99624); colon_aca resize_only: P@1 1.00000, P@5 0.99916 (stored 1.00000 / 0.99916).

| class | features | P@1 | P@5 |
|---|---|---|---|
| colon_n | **Lab histograms (96-d, approach 1 features; this work)** | 0.99360 | 0.97800 |
| colon_n | UNI resize_only (published, stored) | 0.99720 | 0.98516 |
| colon_n | UNI resize_only (my reproduction, stored) | 0.99720 | 0.98508 |
| colon_n | UNI imagenet (published, stored) | 0.99440 | 0.94568 |
| colon_n | UNI imagenet (my reproduction, stored) | 0.99420 | 0.94576 |
| colon_aca | **Lab histograms (96-d, approach 1 features; this work)** | 0.99880 | 0.99076 |
| colon_aca | UNI resize_only (published, stored) | 1.00000 | 0.99920 |
| colon_aca | UNI resize_only (my reproduction, stored) | 1.00000 | 0.99916 |
| colon_aca | UNI imagenet (published, stored) | 0.99920 | 0.99628 |
| colon_aca | UNI imagenet (my reproduction, stored) | 0.99920 | 0.99624 |

## Test A2: controlled rotation

- Selection: `RandomState(0)` draws 100 distinct published groups per class, then one image per group (saved in `testA.csv`, section `selection`).
- Pipeline: original image, PIL `rotate(angle, BILINEAR, expand=False)` (black corners), centre-crop to the largest axis-aligned square without empty corners (side = 768/(cos+sin), floor, minus a 2 px margin, made even), bilinear resize back to 768. Crop sides: 0 deg 768 px, 5 deg 706 px, 10 deg 660 px, 15 deg 624 px, 20 deg 596 px, 25 deg 574 px. The 0 deg condition runs through the same code (crop 768, no change).
- Features, all on the same in-memory image: Lab = approach 1 histogram (raw 96-d); UNI imagenet = the approach 3 / stage 2 timm transform (resize, centre crop, ImageNet normalisation); UNI resize-only = upstream `Resize(224)` on a uint8 tensor, scaled to [0,1], no normalisation.
- Ratio per image = distance from the transformed image's feature to the 0 deg feature of the same image, divided by one denominator per feature and class: the median over the 100 selected images of the distance from the unrotated image to its nearest image of a **different** published group (searched over all 5000 images of the class, same space). Primary space: Lab raw Euclidean; both UNI features L2-normalised Euclidean. Extra columns `raw_euclid_extra`: UNI without L2 normalisation (the reference for imagenet is then the upstream UNI/imagenet features, whose preprocessing differs slightly from the approach 3 transform).
- Also reported: the 90th percentile of the ratio, and the fraction of images whose transformed version is farther from its own 0 deg feature than that image's own nearest different-group image (i.e. the transform moved it past its nearest other original).
- **Control (zoom only).** The crop and resize back magnify the image (by 768/crop, up to 1.34x at 25 deg). `rotation_control_zoom_only` applies the same centre crop and resize to the unrotated image, to separate the effect of the zoom from the effect of the rotation.
- **Horizontal flip** at 0 deg (no crop) is its own row, `condition = hflip`.

### colon_n: median ratio (p90 in brackets; fraction of images moved past their nearest other original)

| condition | Lab | UNI imagenet | UNI resize-only |
|---|---|---|---|
| rotation 0 deg | 0.000 (0.000; 0.00) | 0.000 (0.000; 0.00) | 0.000 (0.000; 0.00) |
| rotation 5 deg | 0.350 (0.526; 0.03) | 0.442 (0.527; 0.00) | 0.447 (0.526; 0.00) |
| rotation 10 deg | 0.479 (0.814; 0.10) | 0.599 (0.699; 0.00) | 0.600 (0.683; 0.00) |
| rotation 15 deg | 0.633 (1.160; 0.15) | 0.763 (0.858; 0.04) | 0.735 (0.850; 0.03) |
| rotation 20 deg | 0.736 (1.418; 0.26) | 0.871 (1.016; 0.22) | 0.844 (0.980; 0.09) |
| rotation 25 deg | 0.886 (1.546; 0.37) | 0.973 (1.112; 0.42) | 0.933 (1.076; 0.33) |
| horizontal flip | 0.000 (0.000; 0.00) | 0.522 (0.765; 0.00) | 0.508 (0.710; 0.00) |

Zoom-only control (same crop, no rotation), median ratio: 5 deg: 0.334 / 0.427 / 0.428; 10 deg: 0.535 / 0.554 / 0.557; 15 deg: 0.667 / 0.665 / 0.675; 20 deg: 0.788 / 0.761 / 0.763; 25 deg: 0.870 / 0.825 / 0.827 (Lab / UNI imagenet / UNI resize-only).

Denominators (median nearest-different-group distance): lab 0.1439, uni_imagenet 0.4389, uni_resize_only 0.3683. Checks, in-session 0 deg feature vs the stored reference row: max abs difference lab 0.00e+00, uni_imagenet 1.60e-07, uni_resize_only 3.59e-07.

### colon_aca: median ratio (p90 in brackets; fraction of images moved past their nearest other original)

| condition | Lab | UNI imagenet | UNI resize-only |
|---|---|---|---|
| rotation 0 deg | 0.000 (0.000; 0.00) | 0.000 (0.000; 0.00) | 0.000 (0.000; 0.00) |
| rotation 5 deg | 0.260 (0.478; 0.00) | 0.372 (0.456; 0.00) | 0.374 (0.461; 0.00) |
| rotation 10 deg | 0.406 (0.779; 0.03) | 0.477 (0.553; 0.00) | 0.486 (0.563; 0.00) |
| rotation 15 deg | 0.511 (1.022; 0.09) | 0.589 (0.678; 0.00) | 0.578 (0.672; 0.00) |
| rotation 20 deg | 0.661 (1.147; 0.18) | 0.676 (0.799; 0.02) | 0.651 (0.787; 0.01) |
| rotation 25 deg | 0.721 (1.281; 0.20) | 0.756 (0.872; 0.02) | 0.710 (0.866; 0.01) |
| horizontal flip | 0.000 (0.000; 0.00) | 0.391 (0.476; 0.00) | 0.374 (0.473; 0.00) |

Zoom-only control (same crop, no rotation), median ratio: 5 deg: 0.236 / 0.365 / 0.370; 10 deg: 0.388 / 0.457 / 0.470; 15 deg: 0.500 / 0.548 / 0.544; 20 deg: 0.563 / 0.624 / 0.621; 25 deg: 0.644 / 0.691 / 0.677 (Lab / UNI imagenet / UNI resize-only).

Denominators (median nearest-different-group distance): lab 0.2081, uni_imagenet 0.5785, uni_resize_only 0.4993. Checks, in-session 0 deg feature vs the stored reference row: max abs difference lab 0.00e+00, uni_imagenet 1.41e-07, uni_resize_only 1.11e-06.

Lab input check: PIL and cv2 decode of the originals give identical histograms (max abs difference colon_n 0.0e+00, colon_aca 0.0e+00); the originals' histograms differ from approach 1's re-saved-image histograms by colon_n mean 1.7e-04 / max 9.0e-03, colon_aca mean 1.7e-04 / max 3.0e-02.

Figure: `testA_rotation.png` (median ratio against angle, three features, horizontal-flip markers at 0 deg, reference line at 1).

## Test B: ORB merges of approach 1

Each logged merge (kept, removed, ORB score) is looked up by the published group of the two representative images (the logged representative file of each piece). Caveat: a merged piece can contain several published groups, so a merge that is correct at representative level can still join impure pieces.

| class | kind | merges | ORB score min / median / max |
|---|---|---|---|
| colon_n | same group | 18 | 42 / 152 / 303 |
| colon_n | different group | 18 | 40 / 44 / 50 |
| colon_aca | same group | 25 | 64 / 156 / 332 |
| colon_aca | different group | 5 | 41 / 48 / 52 |

Merges that would have passed each threshold (correct = same published group), and the ARI of the rebuilt a1 partition. `posthoc` applies only the logged merges at or above the threshold (union-find on the logged merges, as specified). **Extra, real replay:** the ORB pass re-run with that threshold (a merge removes a representative from later pairs, so the result differs from filtering after the fact); at threshold 40 both equal the logged result.

| class | threshold | posthoc: correct / wrong | posthoc ARI | groups | extra, real replay: correct / wrong | replay ARI | groups |
|---|---|---|---|---|---|---|---|
| colon_n | no merges | 0 / 0 | 0.7680 | 279 | n/a | n/a | n/a |
| colon_n | 40 | 18 / 18 | 0.6914 | 243 | 18 / 18 | 0.6914 | 243 |
| colon_n | 50 | 16 / 1 | 0.7895 | 262 | 17 / 4 | 0.7853 | 258 |
| colon_n | 60 | 16 / 0 | 0.7912 | 263 | 18 / 0 | 0.7934 | 261 |
| colon_n | 80 | 15 / 0 | 0.7920 | 264 | 16 / 0 | 0.7929 | 263 |
| colon_n | 100 | 15 / 0 | 0.7920 | 264 | 16 / 0 | 0.7929 | 263 |
| colon_n | 150 | 11 / 0 | 0.7888 | 268 | 12 / 0 | 0.7897 | 267 |
| colon_aca | no merges | 0 / 0 | 0.8207 | 276 | n/a | n/a | n/a |
| colon_aca | 40 | 25 / 5 | 0.8083 | 246 | 25 / 5 | 0.8083 | 246 |
| colon_aca | 50 | 25 / 2 | 0.8264 | 249 | 26 / 2 | 0.8271 | 248 |
| colon_aca | 60 | 25 / 0 | 0.8353 | 251 | 26 / 0 | 0.8360 | 250 |
| colon_aca | 80 | 23 / 0 | 0.8344 | 253 | 24 / 0 | 0.8351 | 252 |
| colon_aca | 100 | 20 / 0 | 0.8310 | 256 | 21 / 0 | 0.8317 | 255 |
| colon_aca | 150 | 15 / 0 | 0.8241 | 261 | 16 / 0 | 0.8248 | 260 |

Checks: the post-hoc rebuild at threshold 40 equals `a1_final` exactly (colon_n: True, colon_aca: True); the replay at threshold 40 equals the logged merge sequence (colon_n: True, colon_aca: True).

## Reading of the numbers

**Test A1.** Lab histograms re-identify their own group almost as well as UNI by P@1/P@5: colon_n P@1 0.9936 / P@5 0.9780 (UNI resize-only, my reproduction: 0.9972 / 0.9851); colon_aca P@1 0.9988 / P@5 0.9908 (UNI resize-only, my reproduction: 1.0000 / 0.9992).

**Test A2.**

- colon_n, 25 deg median ratio (rotation / zoom-only control): Lab 0.886 / 0.870; UNI imagenet 0.973 / 0.825; UNI resize-only 0.933 / 0.827. Fraction of images moved past their nearest other original at 25 deg: Lab 0.37, UNI imagenet 0.42, UNI resize-only 0.33.
- colon_aca, 25 deg median ratio (rotation / zoom-only control): Lab 0.721 / 0.644; UNI imagenet 0.756 / 0.691; UNI resize-only 0.710 / 0.677. Fraction of images moved past their nearest other original at 25 deg: Lab 0.20, UNI imagenet 0.02, UNI resize-only 0.01.
- Zoom-only control: the crop and magnification alone give between 85% (colon_n, UNI imagenet, 25 deg) and 112% (colon_n, Lab, 10 deg) of the median ratio of the rotated images, over all classes, features and angles. So a large part of the displacement measured here comes from the crop and magnification (content removed, scale changed), not from the rotation; the rotation-only part is the difference between the two curves, and it is not separable beyond that with this design (a control rotating by multiples of 90 degrees, with no crop, would isolate it).
- Median ratios stay below 1 for all three features at every angle up to 25 deg in both classes. First angle at which the 90th percentile of the ratio reaches 1: colon_n Lab: from 15 deg; colon_n UNI imagenet: from 20 deg; colon_n UNI resize-only: from 25 deg; colon_aca Lab: from 15 deg; colon_aca UNI imagenet: never up to 25 deg; colon_aca UNI resize-only: never up to 25 deg.
- Horizontal flip, colon_n: median ratio Lab 0.000, UNI imagenet 0.522, UNI resize-only 0.508 (a colour histogram does not change under a flip, UNI embeddings do).
- Horizontal flip, colon_aca: median ratio Lab 0.000, UNI imagenet 0.391, UNI resize-only 0.374 (a colour histogram does not change under a flip, UNI embeddings do).

**Test B.**

- colon_n: 18 merges join the same published group (ORB score 42 to 303) and 18 join different groups (score 40 to 50). At threshold 60 no wrong merge passes (post-hoc 16 correct / 0 wrong); ARI 0.6914 at the logged threshold 40, 0.7912 post-hoc at 60 (0.7934 with a real replay), 0.7680 with no merges.
- colon_aca: 25 merges join the same published group (ORB score 64 to 332) and 5 join different groups (score 41 to 52). At threshold 60 no wrong merge passes (post-hoc 25 correct / 0 wrong); ARI 0.8083 at the logged threshold 40, 0.8353 post-hoc at 60 (0.8360 with a real replay), 0.8207 with no merges.

