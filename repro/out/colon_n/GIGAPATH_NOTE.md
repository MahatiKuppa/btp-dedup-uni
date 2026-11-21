# Prov-GigaPath row for colon_n

New file. No existing colon_n output was changed: `repro/out/phase3_compare.csv` and the other 11-combo outputs still hold 11 combos.
The row below was evaluated in a scratch folder (upstream `eval_results/` was not written) and is also in `phase3_gigapath_row.csv` and `phase3_gigapath_eval.*` next to this note.
The side-by-side summary with colon_aca (11 and 12 combos) is in `repro/REPORT_colon_aca.md`.

## Provenance and caveat

These features were extracted on a **Colab T4 GPU** (GPU: Tesla T4 | torch 2.14.1+cu130 | timm 1.0.31 | fp32 | batch 32 | upstream 912f342fe2a781e847f395d79982dad4d3d425ae | zip sha256 matched for both classes),
not on the CPU like the other 11 combos (local `venv_repro`: torch 2.14.1+cpu, timm 1.0.30, CPU kernels). Last float digits can therefore differ; the six sha256 lines are in `repro/gigapath_colab_sha256.txt`.

## prov-gigapath#imagenet, UNI-resize_only annotations, NoReduction, euclidean, KMeans, k = 257

| metric | theirs | mine | delta | bar | result |
|---|---|---|---|---|---|
| P@1 | 0.99700 | 0.99700 | +0.00000 | 0.002 | PASS |
| P@5 | 0.98356 | 0.98356 | +0.00000 | 0.002 | PASS |
| ARI | 0.50260 | 0.49332 | -0.00928 | 0.005 | FAIL |
| FMI | 0.50560 | 0.49642 | -0.00918 | 0.005 | FAIL |

All four bars: FAIL. KMeans seed spread (seeds 0..9, `n_init` default) on these features: ARI 0.5105 ± 0.0120 [0.4934, 0.5287]; their ARI is z = -0.66 from that mean (inside 2 sd).
