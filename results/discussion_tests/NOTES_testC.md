# Test C: why Lab histograms beat UNI resize_only under KMeans on colon_n

Question: on colon_n, 96-d Lab histograms (approach 1/2 KMeans, ARI 0.73) beat UNI resize_only (benchmark ARI 0.64, 10-start mean 0.68) although UNI has the higher P@5. Run on colon_n and colon_aca.

Script: `scripts/discussion_testc.py` (steps t1, t2, t3, report), sklearn 1.9.1 and opencv 5.1.0 (`data/approach12_drive/venv_a12`), CPU. Feature sets: (a) the approach12 Lab histograms (computed from the re-saved images), (b) and (c) the stored upstream UNI `resize_only` and `imagenet` features in `third_party/.../feature_vectors` (raw), each raw and L2-normalised. ARI is against the published groups (257 for colon_n, 249 for colon_aca).

**No setting was chosen by looking at ARI.** Every method, linkage, `n_neighbors` value and seed below was fixed in the plan before any result was seen, and all of them are reported (every spectral `n_neighbors` x seed run, every linkage), with the range over all settings given instead of a best one. Runs that failed or warned are listed as such, not dropped.

## Reference values already in the repro (not recomputed here)

- colon_n, UNI resize_only raw, KMeans (k = group count), seeds 0..9: `n_init='auto'` (= 1 start for k-means++) mean 0.6628 ± 0.0110; `n_init=10` mean 0.6781 ± 0.0156. The benchmark number is seed 42 with `n_init='auto'`: 0.6434.
- colon_aca, UNI resize_only raw, KMeans (k = group count), seeds 0..9: `n_init='auto'` (= 1 start for k-means++) mean 0.9110 ± 0.0143; `n_init=10` mean 0.9274 ± 0.0068. The benchmark number is seed 42 with `n_init='auto'`: 0.9232.
- Approach 1/2 plain Lab KMeans, k=250, seed 42, `n_init=10`, as Colab ran it (sklearn 1.7.2, which reproduces Colab's labels; see `results/approach12_NOTES.md`): colon_n ARI 0.7322, colon_aca 0.7857.

## T1: KMeans with k = published group count, `n_init=10`, seeds 0..9

### colon_n

| feature set | k | ARI mean ± sd | min / max |
|---|---|---|---|
| (a) Lab histograms, raw | 257 | 0.7192 ± 0.0123 | 0.7065 / 0.7490 |
| (a) Lab histograms, L2-normalised | 257 | 0.7224 ± 0.0087 | 0.7065 / 0.7366 |
| (b) UNI resize_only, raw | 257 | 0.6781 ± 0.0156 | 0.6519 / 0.6944 |
| (b) UNI resize_only, L2 | 257 | 0.6708 ± 0.0157 | 0.6439 / 0.6984 |
| (c) UNI imagenet, raw | 257 | 0.4634 ± 0.0134 | 0.4410 / 0.4819 |
| (c) UNI imagenet, L2 | 257 | 0.4615 ± 0.0147 | 0.4397 / 0.4825 |
| (a) Lab histograms, raw, **k=250** | 250 | 0.7118 ± 0.0120 | 0.6885 / 0.7313 |

Lab raw, k=250, seed 42, `n_init=10` under sklearn 1.9.1: 0.7388; the Colab run (sklearn 1.7.2) scored 0.7322 (for colon_n the two sklearn versions give different partitions, for colon_aca the same one).
Cross-check: T1 UNI resize_only raw (`n_init=10`, seeds 0..9) vs the stored repro run: max abs difference 0.00e+00.
Lab raw vs L2-normalised: every Lab row has norm 1.732051 to 1.732051 (sqrt(3) = 1.732051), so the two inputs are the same up to a constant factor. Even so, individual seeds differ by up to 0.034 ARI between them, comparable to the seed-to-seed sd above, so differences of that size between two KMeans runs on essentially the same features are run-to-run noise, not feature effects.

### colon_aca

| feature set | k | ARI mean ± sd | min / max |
|---|---|---|---|
| (a) Lab histograms, raw | 249 | 0.7929 ± 0.0080 | 0.7770 / 0.8052 |
| (a) Lab histograms, L2-normalised | 249 | 0.7892 ± 0.0120 | 0.7699 / 0.8055 |
| (b) UNI resize_only, raw | 249 | 0.9274 ± 0.0068 | 0.9196 / 0.9388 |
| (b) UNI resize_only, L2 | 249 | 0.9291 ± 0.0110 | 0.9160 / 0.9535 |
| (c) UNI imagenet, raw | 249 | 0.8595 ± 0.0139 | 0.8385 / 0.8872 |
| (c) UNI imagenet, L2 | 249 | 0.8630 ± 0.0183 | 0.8341 / 0.8951 |
| (a) Lab histograms, raw, **k=250** | 250 | 0.7831 ± 0.0088 | 0.7709 / 0.7976 |

Lab raw, k=250, seed 42, `n_init=10` under sklearn 1.9.1: 0.7857; the Colab run (sklearn 1.7.2) scored 0.7857 (for colon_n the two sklearn versions give different partitions, for colon_aca the same one).
Cross-check: T1 UNI resize_only raw (`n_init=10`, seeds 0..9) vs the stored repro run: max abs difference 0.00e+00.
Lab raw vs L2-normalised: every Lab row has norm 1.732051 to 1.732051 (sqrt(3) = 1.732051), so the two inputs are the same up to a constant factor. Even so, individual seeds differ by up to 0.029 ARI between them, comparable to the seed-to-seed sd above, so differences of that size between two KMeans runs on essentially the same features are run-to-run noise, not feature effects.

## T2: other clustering methods on the L2-normalised feature sets (k = group count)

Agglomerative: one run per linkage (deterministic). Spectral (`affinity='nearest_neighbors'`): `n_neighbors` 5, 10, 15, 20, 30, seeds 0, 1, 2. All settings are shown. `W` = sklearn warned that the neighbour graph is not fully connected or a solver did not converge in at least one run of that setting.

### colon_n

| setting | (a) Lab | (b) UNI resize_only | (c) UNI imagenet |
|---|---|---|---|
| agglomerative average linkage | 0.6243 | 0.3723 | 0.1926 |
| agglomerative complete linkage | 0.5995 | 0.3948 | 0.2897 |
| agglomerative single linkage | 0.2564 | 0.1062 | 0.0390 |
| agglomerative ward linkage | 0.7599 | 0.7646 | 0.5421 |
| spectral 5 neighbours | 0.7987 ± 0.0017 W | 0.9101 ± 0.0078 W | 0.5086 ± 0.0213 W |
| spectral 10 neighbours | 0.8324 ± 0.0033 W | 0.8737 ± 0.0034 W | 0.5692 ± 0.0123 W |
| spectral 15 neighbours | 0.8213 ± 0.0054 W | 0.8426 ± 0.0036 W | 0.5863 ± 0.0011 W |
| spectral 20 neighbours | 0.8152 ± 0.0060 W | 0.8258 ± 0.0095 W | 0.5848 ± 0.0151 |
| spectral 30 neighbours | 0.7574 ± 0.0040 W | 0.7443 ± 0.0084 | 0.5264 ± 0.0099 |

Range over all settings of a family (min to max ARI over every run): Lab agglomerative 0.2564 to 0.7599; Lab spectral 0.7531 to 0.8348; UNI resize_only agglomerative 0.1062 to 0.7646; UNI resize_only spectral 0.7348 to 0.9190; UNI imagenet agglomerative 0.0390 to 0.5421; UNI imagenet spectral 0.4942 to 0.5977.

### colon_aca

| setting | (a) Lab | (b) UNI resize_only | (c) UNI imagenet |
|---|---|---|---|
| agglomerative average linkage | 0.7030 | 0.8813 | 0.8328 |
| agglomerative complete linkage | 0.6559 | 0.7512 | 0.6901 |
| agglomerative single linkage | 0.6952 | 0.9188 | 0.7925 |
| agglomerative ward linkage | 0.7929 | 0.9733 | 0.9526 |
| spectral 5 neighbours | 0.7051 ± 0.0000 W | 0.9764 ± 0.0033 W | 0.1298 ± 0.0005 W |
| spectral 10 neighbours | 0.9041 ± 0.0029 W | 0.9845 ± 0.0000 W | 0.9237 ± 0.0005 W |
| spectral 15 neighbours | 0.8763 ± 0.0028 W | 0.9721 ± 0.0024 W | 0.9378 ± 0.0013 W |
| spectral 20 neighbours | 0.8538 ± 0.0047 W | 0.9717 ± 0.0012 W | 0.9190 ± 0.0057 W |
| spectral 30 neighbours | 0.8116 ± 0.0052 W | 0.9132 ± 0.0038 W | 0.8283 ± 0.0078 W |

Range over all settings of a family (min to max ARI over every run): Lab agglomerative 0.6559 to 0.7929; Lab spectral 0.7051 to 0.9070; UNI resize_only agglomerative 0.7512 to 0.9733; UNI resize_only spectral 0.9105 to 0.9845; UNI imagenet agglomerative 0.6901 to 0.9526; UNI imagenet spectral 0.1295 to 0.9390.

## T3: colour shortcut

Reinhard normalisation: each original image is converted to 8-bit Lab with cv2 (as approach 1), and each channel is rescaled so that its mean and std equal the class-wide averages of the per-image means and stds, clipped to [0, 255] and rounded; the 32-bin histograms are computed directly on these Lab channels (no round trip through BGR). `lab_originals` is the same histogram computed from the original files (approach 1's features come from the re-saved images), as a control. `moments6` are the per-image mean and std of the three 8-bit Lab channels (6 numbers; raw, and z-scored per dimension as an extra). KMeans: k = group count, `n_init=10`, seeds 0..9. P@1/P@5: upstream definition, Euclidean on the features as given.

### colon_n

Targets (class-wide average per-image Lab mean / std): mean 199.5, 149.5, 106.7, std 40.6, 14.1, 17.0. After normalisation and clipping the per-image mean is within 0.69 (mean abs 0.050) and the std within 0.91 (mean abs 0.057) of the targets; 2969 of 5000 images had any clipped pixel (mean clipped fraction 5.21e-03). Originals vs re-saved histograms: max abs difference 0.009.

| feature set | P@1 | P@5 | KMeans ARI mean ± sd | min / max |
|---|---|---|---|---|
| lab_approach12_resaved | 0.9936 | 0.9780 | 0.7192 ± 0.0123 | 0.7065 / 0.7490 |
| lab_originals | 0.9936 | 0.9782 | 0.7245 ± 0.0086 | 0.7102 / 0.7385 |
| lab_reinhard | 0.9142 | 0.7733 | 0.4443 ± 0.0080 | 0.4343 / 0.4576 |
| moments6_raw | 0.9538 | 0.8277 | 0.4012 ± 0.0045 | 0.3936 / 0.4090 |
| moments6_zscored | 0.9746 | 0.8939 | 0.5194 ± 0.0101 | 0.5001 / 0.5335 |

P@1 / P@5 reference rows (same function): (a) Lab histograms, raw 0.9936 / 0.9780; (a) Lab histograms, L2-normalised 0.9936 / 0.9780; (b) UNI resize_only, raw 0.9972 / 0.9851; (b) UNI resize_only, L2 0.9972 / 0.9852; (c) UNI imagenet, raw 0.9942 / 0.9458; (c) UNI imagenet, L2 0.9944 / 0.9461.

### colon_aca

Targets (class-wide average per-image Lab mean / std): mean 192.4, 150.1, 102.3, std 36.7, 11.5, 15.2. After normalisation and clipping the per-image mean is within 1.69 (mean abs 0.082) and the std within 3.88 (mean abs 0.129) of the targets; 3183 of 5000 images had any clipped pixel (mean clipped fraction 7.15e-03). Originals vs re-saved histograms: max abs difference 0.030.

| feature set | P@1 | P@5 | KMeans ARI mean ± sd | min / max |
|---|---|---|---|---|
| lab_approach12_resaved | 0.9988 | 0.9908 | 0.7929 ± 0.0080 | 0.7770 / 0.8052 |
| lab_originals | 0.9988 | 0.9906 | 0.7874 ± 0.0097 | 0.7675 / 0.7987 |
| lab_reinhard | 0.9582 | 0.8508 | 0.5074 ± 0.0058 | 0.4956 / 0.5164 |
| moments6_raw | 0.9774 | 0.9015 | 0.5718 ± 0.0082 | 0.5586 / 0.5877 |
| moments6_zscored | 0.9832 | 0.9334 | 0.6274 ± 0.0080 | 0.6130 / 0.6440 |

P@1 / P@5 reference rows (same function): (a) Lab histograms, raw 0.9988 / 0.9908; (a) Lab histograms, L2-normalised 0.9988 / 0.9908; (b) UNI resize_only, raw 1.0000 / 0.9992; (b) UNI resize_only, L2 1.0000 / 0.9992; (c) UNI imagenet, raw 0.9992 / 0.9962; (c) UNI imagenet, L2 0.9992 / 0.9961.

T3b (UNI resize_only on the Reinhard-normalised images) was not run: it needs about 38 minutes of UNI extraction per class and was left to the user's decision.

## Reading of the numbers

**T1 (KMeans, `n_init=10`).**
- colon_n: Lab minus UNI resize_only is +0.041 (raw) and +0.052 (L2) ARI; seed sd is 0.016 and 0.016. UNI imagenet: 0.463 raw, 0.462 L2.
- colon_aca: Lab minus UNI resize_only is -0.134 (raw) and -0.140 (L2) ARI; seed sd is 0.008 and 0.012. UNI imagenet: 0.860 raw, 0.863 L2.
- So the ordering Lab > UNI resize_only under KMeans holds on colon_n and reverses on colon_aca, where UNI is far ahead; it is not a general property of the two feature types.
- `n_init`: for UNI resize_only raw on colon_n the mean over seeds 0..9 is 0.663 with `n_init='auto'` and 0.678 with `n_init=10` (reference values above). `n_init=10` therefore moves the seed mean up by about 0.015; the benchmark's 0.643 (seed 42, `n_init='auto'`) sits at the bottom of the `n_init='auto'` seed range. It does not close the gap to Lab (about 0.72).

**T2 (method).**
- colon_n, mean ARI of UNI resize_only above Lab in 5 of 9 settings (agglomerative ward, spectral 5, spectral 10, spectral 15, spectral 20); Lab above UNI in agglomerative average, agglomerative complete, agglomerative single, spectral 30.
- colon_n, range over all settings: spectral Lab 0.753 to 0.835, spectral UNI resize_only 0.735 to 0.919, spectral UNI imagenet 0.494 to 0.598; KMeans (T1, L2) Lab 0.722, UNI resize_only 0.671.
- colon_aca, mean ARI of UNI resize_only above Lab in 9 of 9 settings (agglomerative average, agglomerative complete, agglomerative single, agglomerative ward, spectral 5, spectral 10, spectral 15, spectral 20, spectral 30); Lab above UNI in none.
- colon_aca, range over all settings: spectral Lab 0.705 to 0.907, spectral UNI resize_only 0.911 to 0.985, spectral UNI imagenet 0.129 to 0.939; KMeans (T1, L2) Lab 0.789, UNI resize_only 0.929.
- On colon_n, spectral clustering on the same UNI resize_only features is above KMeans (L2, `n_init=10`) in every `n_neighbors` setting tried, by 0.073 to 0.239 ARI, while average, complete and single linkage are far below KMeans (0.372, 0.395, 0.106 against 0.671). The ranking of the feature sets therefore depends on the clustering method, not only on the features. Spectral clustering warned about a not fully connected neighbour graph in most settings (marked `W`), and the colon_aca UNI imagenet result at 5 neighbours (about 0.13) shows what a failed graph looks like; no setting was removed or selected.

**T3 (colour shortcut).**
- colon_n: equalising each image's Lab channel mean and std lowers Lab KMeans ARI from 0.725 to 0.444 and P@1/P@5 from 0.9936 / 0.9782 to 0.9142 / 0.7733. The six mean/std numbers alone give P@1/P@5 0.9538 / 0.8277 and KMeans ARI 0.401 (z-scored: 0.9746 / 0.8939, ARI 0.519).
- colon_aca: equalising each image's Lab channel mean and std lowers Lab KMeans ARI from 0.787 to 0.507 and P@1/P@5 from 0.9988 / 0.9906 to 0.9582 / 0.8508. The six mean/std numbers alone give P@1/P@5 0.9774 / 0.9015 and KMeans ARI 0.572 (z-scored: 0.9832 / 0.9334, ARI 0.627).
- Per-image colour mean and std carry a large part of the grouping information in the Lab histograms (six numbers reach P@1 of 0.95 to 0.98, against 0.99 to 1.00 for the full histogram), and much of the Lab histogram's KMeans ARI disappears when they are equalised. Two limits: the normalisation also stretches and clips histograms (about 3000 of 5000 images have some clipped pixels), so part of the drop may come from that distortion and not only from removing the colour moments; and these tests do not show why UNI does worse under KMeans. Whether UNI also loses ARI on colour-equalised images is the T3b question (UNI resize_only on the Reinhard images), not run.

