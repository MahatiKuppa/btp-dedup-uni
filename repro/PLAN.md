# Reproduce Batchkala et al. (LC25000-clean) on colon_n

Approved by user. No Co-Authored-By trailer on commits (user instruction). Run phase by phase; stop after each phase with: status report, flags, open questions, next steps.

## Context
Goal: run upstream code AS IS on colon_n (5000 imgs), and analyse cluster size distributions (their final 257, their initial 250, my initial 250, my benchmark run). Check if my clusters and benchmark numbers match theirs, and find the stage where divergence starts. Upstream facts verified by reading the repo (no clone yet):

- `source/constants.py`: `PROJECT_PATH` = repo root (resolved from file location). `RANDOM_SEED=42`. Features read from `<repo>/feature_vectors/<cancer>/<extractor>/<norm>/`. Annotations read from `<repo>/annotations/<cancer>/UNI/resize_only` (hardcoded unless `--manual_annotations_dir`).
- `eval_utils.reduce_feature_dimensionality`: `umap.UMAP(n_components, random_state=42)`, no n_jobs. `get_clustering_labels`: `KMeans(n_clusters, random_state)`, **n_init NOT set, so sklearn default applies** (10 in <1.4, 'auto'=1 for k-means++ in >=1.4). Big reproducibility risk: their .pkl `_sklearn_version`/`n_init` tell which one they used.
- `evaluate_clustering.py`: k = `len(final_clusters.json)` (257). Output is fixed name `eval_results/cancer_type=colon_n#extractor_name=all#...=all.{json,csv}` relative to cwd. Skips combos already in the JSON unless `--overwrite`. Does not create `eval_results/`.
- `extract_features.py` **device bug**: `device = args.device if cuda available else 'cpu'`, but raw `args.device` goes to `prepare_feature_extractor`; non-cpu without GPUs raises `NotImplementedError`. With `--device cpu` the model path is fine; input `.to(device)` uses the fallback var = cpu. Likely no patch needed. `--precision half` is autocast fp16: do not use. Prompts on stdin before overwriting differing features.
- Model loading (`source/feature_extraction/get_model.py`): UNI `hf-hub:MahmoodLab/UNI` via timm; prov-gigapath `hf_hub:prov-gigapath/prov-gigapath` (asserts timm>1.0.0); DINOv2 via torch.hub `facebookresearch/dinov2`; ResNet18 torchvision `IMAGENET1K_V1`; ResNet50-CLAM `resnet50_baseline(pretrained=True)`; phikon wrapper; SimCLR via `get_resnet18_dsmil(weights_path=EXTRACTOR_NAMES_2_WEIGHTS_PATHS[name])` (paths defined in constants; read after clone).
- Their `.gitignore` probably ignores `feature_vectors/`; check after clone. Writing feature_vectors/ into their tree is allowed (not annotations/ or eval_results/).

Machine: WSL2, 20 cores, 7 GB total / ~5 GB free RAM, 918 GB disk, Python 3.12.3 system, `uv` available at `~/.local/bin/uv`. Git: `data/`, `venv/`, `.env` already ignored. Need to add `venv_repro/`, `third_party/` handling (see below) to `.gitignore`.

## Decisions I will make (flag if you disagree)
- `third_party/LC25000-clean` is a plain `git clone` (own `.git`); add `third_party/` to my `.gitignore`. Record the commit hash in `repro/UPSTREAM_COMMIT.txt` (committed). Feature_vectors there are not committed; copies of small outputs go to `repro/out/`.
- venv via `uv venv --python 3.12 venv_repro`; `uv pip install` with retries (loop 5x, `UV_HTTP_TIMEOUT=120`). torch/torchvision from the CPU index (`https://download.pytorch.org/whl/cpu`).
- All repro scripts live in `repro/`, run with `PYTHONPATH=third_party/LC25000-clean` so `import source.eval_utils` works. No edits in upstream tree unless unavoidable; each edit logged in `repro/PATCHES.md` with diff.
- Tokens only read from `.env` into env var `HF_TOKEN`; never printed or committed.

## PHASE 0: setup (~15 min, <2 GB RAM)
1. `git clone` upstream, record hash. Read `environment-creation.md`, `source/constants.py` (weight paths), notebook 2 + 3 code cells (for exact Phase 1/2 logic).
2. Create `venv_repro`, install: torch, torchvision (CPU), timm (latest >=1.0.3), scikit-learn, umap-learn, transformers, pandas, matplotlib, tqdm, scipy, huggingface_hub, jupyter-free (nbformat only if needed to read notebooks). `pip freeze` and key versions to `repro/env_versions.txt`.
3. Data: `ln -s ~/btp_dedup_uni/data/raw_tampapath/colon_n third_party/LC25000-clean/LC25000/colon_n`. Check 5000 `colonn*.jpeg`; check every path in `final_clusters.json` exists. (Existing project listing shows `data/raw_tampapath/colon_n`; verify it holds the jpegs.)
4. HF access check via `huggingface_hub` API with the `.env` token: `MahmoodLab/UNI`, `owkin/phikon`, `prov-gigapath/prov-gigapath`. **If gigapath denied: STOP and tell you to request access.**
5. Weights status table (UNI, phikon, gigapath, dinov2 S/B via torch.hub dry load, resnet18, resnet50-clam, simclr 2.5x/10x). SimCLR: try fetch from dsmil-wsi links; if fail **STOP** with exact path `weights/simclr-tcga-lung/weights-{2.5x,10x}/model-v1.pth` (relative to upstream root, confirm from constants).
6. Smoke test: extract 16 images on cpu to confirm no patch needed; log in `repro/PATCHES.md` (even if "none").
Files: `repro/PLAN.md, UPSTREAM_COMMIT.txt, env_versions.txt, PATCHES.md, weights_status.txt`. Commit.
Risks: PyPI flaky (retry loop); umap/numba on py3.12 wheels; torch.hub needs GitHub reachability; ResNet50-CLAM helper may fetch weights from a URL.

## PHASE 1: initial clustering (~40 min, ~3 GB RAM)
1. `python extract_features.py --cancer_type colon_n --img_norm resize_only --extractor_name UNI --device cpu --batch_size 16`. Est. 15-30 min (ViT-L/16, 5000 imgs, 20 threads). Run in background with `nohup`, log to `repro/logs/`. Not resumable internally (single save at end): if it dies, rerun; mitigate by smoke test first.
2. `repro/stage_a_initial_clustering.py`: imports `source.eval_utils`; features -> `reduce_feature_dimensionality(..., 'UMAP-8')` -> `get_clustering_labels(n_clusters=250, 'kmeans', random_state=42)` -> `get_clustering_centroids` -> anchor = member nearest centroid (as in notebook 2; exact distance/tie logic copied from notebook). Save `labels.npy, centers.npy, anchors.json` to `repro/out/initial/`.
3. Load their `annotations/colon_n/UNI/resize_only/*.pkl`; print `_sklearn_version`, `n_init`, `n_features_in_`, cluster sizes. Note: n_features_in_ should be 8 (UMAP space); my installed sklearn version vs theirs is the first divergence suspect.
4. Compare mine vs theirs: label agreement through ids_2_img_paths ordering (map by image filename, not index), ARI, AMI, Hungarian-matched identical-cluster fraction, # of my 250 anchors in their `results_processed.json` keys, size distributions side by side.
5. Noise floor: seeds 0..9 UMAP+KMeans; mean±sd pairwise ARI among mine, mean±sd ARI of each vs their .pkl. ~10 UMAP runs x 1-2 min.
Outputs: `repro/out/initial/*`, `repro/out/phase1_metrics.json`, logs. Commit scripts + small outputs (not features).
Risks: UMAP is not bitwise portable across umap/numba/pynndescent versions even with same seed, so exact match unlikely; that is a finding, not a bug. UMAP on 1024-d x5000 RAM small.

## PHASE 2: rebuild final groups from recorded decisions (~5 min, trivial RAM)
1. Script `repro/stage_b_rebuild_groups.py`. Inputs (read-only): their .pkl labels, `results_processed.json` (anchor + belonging), `pure_rejected_clusters.json`, `final_clusters.json`. Schema inspected first; reconstruct pre-merge set exactly as notebook 2 does (accepted ids 0..249, rejected 250+). Assert every image appears exactly once.
2. Diff vs `final_clusters.json`: infer merges (pre-merge clusters whose images union into one final group), count, list; fraction of the 257 final groups equal to one initial cluster unchanged; rejection rate per cluster.
Outputs: `repro/out/phase2_*.json/csv`. Commit.
Risks: notebook may order/renumber clusters differently; I will follow notebook cells literally. Image-key format (relative path vs filename) mismatch; normalise.

## PHASE 3: benchmark (hours; run in background, checkpointed)
Extraction order, cheapest first, each combo skipped if `features.npy` exists (driver script `repro/run_extractions.sh` with `--device cpu --batch_size 16-32`, log per combo, `yes n |` not used; script checks existence itself so no stdin prompt):
| # | combo | est. time | RAM |
|---|---|---|---|
| 1-2 | resnet18 resize_only, imagenet | 2-4 min each | <1 GB |
| 3-4 | simclr 2.5x, 10x (imagenet norm) | 2-4 min each | <1 GB |
| 5 | resnet50-clam imagenet | 5-8 | ~1 GB |
| 6 | dinov2_vits14 imagenet | 3-6 | ~1 GB |
| 7-8 | owkin-phikon resize_only, imagenet | 8-15 each | ~1.5 GB |
| 9 | dinov2_vitb14 imagenet | 8-15 | ~1.5 GB |
| 10 | UNI imagenet (resize_only done in Ph.1) | 15-30 | ~2.5 GB |
| 11 | prov-gigapath imagenet, LAST | see below | see below |
(Numbers are rough guesses; I will time the 16-image smoke test per model and report revised estimates before the long run.)

**prov-gigapath**: ViT-giant/14, ~1.1B params. fp32 weights ~4.5 GB + activations at batch 4 => ~5.5-6 GB peak vs ~5 GB free: **high OOM risk**. Compute ~4-5x UNI, likely 1.5-3 h. Before running I will measure free RAM and ask you. Options then: (a) run alone with swap/`.wslconfig` memory raise, (b) bf16 weights (deviates from "as is", numbers may drift slightly), (c) skip and report 11/12.

Eval: scratch dir `repro/scratch_eval/` with empty `eval_results/` (gitignored, outputs copied to `repro/out/phase3/`); `cd` there, run `python <upstream>/evaluate_clustering.py --cancer_type colon_n --distance_metric euclidean --dimensionality_reduction NoReduction --clustering kmeans` (extractor/norm left `all` would hit all dirs under feature_vectors; fine since only my 12 exist; confirm no stray dirs). Remember combos of extractor x norm that I did not extract (e.g. UNI/phikon/resnet18 both norms but others only imagenet) simply do not exist, so they are not evaluated. k=257 from annotations.
Compare: join with committed `eval_results/cancer_type=colon_n#...csv` on the 12 combo keys; deltas for ARI, FMI, NMI, homogeneity, completeness, P@1, P@5 -> `repro/out/phase3_compare.csv`.
Extras (own wrapper `repro/stage_c_extras.py`, imports their metric functions, no edits to their code), UNI resize_only: n_init=10 with seeds 0..9 (mean±sd), k=250 vs 257 (k=250 scored against their 257 truth labels; same metrics), and re-score my 4 old arms (`embeddings/uni1_full.npy, uni1_pca128.npy, resnet50_full.npy, resnet50_pca128.npy`, reordered via `embeddings/paths.txt` / `resnet50_paths.txt` to their ids order) with their metric functions. Checks: path/ordering mapping asserted by filename; existing `venv` not used.
Commit after extraction+eval, and after extras.
Risks: eval result for combos depends on sklearn n_init default (see context); P@k on 5000x5000 cheap; GPU-vs-CPU float differences give tiny deltas; gigapath/ResNet50-CLAM weight downloads; WSL OOM kills the background job (checkpoint = per-combo files; driver resumes).

## PHASE 4: report (~10 min)
`repro/REPORT.md` + `repro/figs/`: tables (Phase 1 agreement vs noise floor, Phase 2 reconstruction, Phase 3 mine vs theirs); histograms and sorted-size curves for: their 257 final, their initial 250, my initial 250, my benchmark KMeans run (k=257). Verdict: reproduced clusters? numbers? first stage where divergence appears (features, UMAP, KMeans/n_init, sklearn version, or annotations). Commit.

## Verification
- Each phase: script exit 0, assertions (5000 images, each exactly once, ids mapping by filename), upstream tree clean for `annotations/` and `eval_results/` (`git -C third_party/LC25000-clean status --porcelain annotations eval_results` empty) checked at the end of every phase.
- Sanity: eval run on my UNI resize_only should have same row structure as theirs.

## Open questions
- Gigapath RAM option (decide at Phase 3, after measurement).
