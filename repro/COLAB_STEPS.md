# Prov-GigaPath on Colab: manual steps

Why: fp32 load needs ~9 GB RAM, this machine has ~6 GB available (see `weights_status.md`).
Output of this procedure replaces `feature_vectors/colon_n/prov-gigapath/imagenet/` locally.

## 1. Upload the zip to Google Drive
1. In Drive create the folder `MyDrive/btp_repro/`.
2. Upload `~/btp_dedup_uni/data/colon_n_original.zip` (471 MB) to it, so the path is `MyDrive/btp_repro/colon_n_original.zip`.
   From WSL the file is at `\\wsl$\<distro>\home\mahati\btp_dedup_uni\data\colon_n_original.zip` in Windows Explorer.
3. Optional integrity check: the sha256 should be `14724d148ccd08631c8977a91e4ef1d5871e959831d29997832ba15bc240169d`
   (`sha256sum` locally; not checked by the notebook).

## 2. Upload the notebook
Open https://colab.research.google.com > File > Upload notebook > `~/btp_dedup_uni/repro/colab_gigapath.ipynb`.

## 3. HF token as a Colab secret
1. Left sidebar key icon (Secrets) > Add new secret.
2. Name `HF_TOKEN`, value = the token from your `.env` (the account that was granted prov-gigapath access).
3. Switch **Notebook access** ON for this notebook.

## 4. Pick the GPU
Runtime > Change runtime type > **T4 GPU** > Save.

## 5. Run
Runtime > Run all. Expect:
- a Drive permission popup (allow);
- ~5 min setup (clone, pip, unzip), then the extraction cell (ViT-giant, 157 batches of 32, roughly 10-25 min on a T4; a progress bar is shown);
- the last cell prints `features: (5000, 1536) float32`, GPU/torch/timm versions and three sha256 lines.
If it fails with CUDA OOM, lower `--batch_size` in cell 6 to 16 and re-run from that cell (everything before is cached in the runtime).
If the runtime disconnects, just Run all again.

## 6. Bring results back
1. In Drive, download `MyDrive/btp_repro/gigapath_out/` (3 files: `features.npy`, `ids.npy`, `ids_2_img_paths.json`).
2. Put them at `~/btp_dedup_uni/third_party/LC25000-clean/feature_vectors/colon_n/prov-gigapath/imagenet/`
   (create the directory; e.g. copy via `\\wsl$\...` or `cp`).
3. Paste the three sha256 lines + GPU/torch/timm line from the notebook output to me. I will verify them against the local files, check that
   `ids_2_img_paths.json` matches the other extractors' ordering, and then include prov-gigapath in the Phase 3 evaluation.

## Notes
- Upstream code is unmodified (pinned commit 912f342); fp32; no `--precision half`.
- GPU features will differ from a CPU run in the last float bits; any ARI/P@k difference from that is a reportable caveat.
