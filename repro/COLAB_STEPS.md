# Prov-GigaPath on Colab: manual steps (colon_n and colon_aca)

Why: fp32 load needs ~9 GB RAM, this machine has ~6 GB available (see `weights_status.md`).
Output of this procedure creates `feature_vectors/<cls>/prov-gigapath/imagenet/` locally for `<cls>` in `colon_n`, `colon_aca`.
Neither class has these features yet.

## 1. Upload the zips to Google Drive
1. In Drive create the folder `MyDrive/btp_repro/`.
2. Upload both zips into it, so the paths are `MyDrive/btp_repro/colon_n_original.zip` and `MyDrive/btp_repro/colon_aca_original.zip`.
   Skip a zip that is already there with the right sha256.
   From Windows Explorer the files are at:

   | class | Windows path | bytes | sha256 |
   |---|---|---|---|
   | colon_n | `\\wsl$\Ubuntu\home\mahati\btp_dedup_uni\data\colon_n_original.zip` | 471608147 | `14724d148ccd08631c8977a91e4ef1d5871e959831d29997832ba15bc240169d` |
   | colon_aca | `\\wsl$\Ubuntu\home\mahati\btp_dedup_uni\data\colon_aca_original.zip` | 489468169 | `830207449b325b5ae80b8ac6575f182df518c58210ac8a9eabe155f3ef31789d` |

   Both zips are flat, stored (no compression), and hold exactly the 5000 original LC25000 files (`colonn*.jpeg`, `colonca*.jpeg`).
   They were built with `cd data/raw_tampapath/<cls> && zip -0 -q ../../<cls>_original.zip *.jpeg`. The colon_aca source files match the CRCs in the official `LC25000.zip`.
3. The notebook checks each zip's sha256 on Colab against the values above and stops on a mismatch (bad or stale upload).
   To check locally first: `sha256sum data/colon_n_original.zip data/colon_aca_original.zip`.

## 2. Upload the notebook
Open https://colab.research.google.com > File > Upload notebook > `~/btp_dedup_uni/repro/colab_gigapath.ipynb`
(`\\wsl$\Ubuntu\home\mahati\btp_dedup_uni\repro\colab_gigapath.ipynb`).

## 3. HF token as a Colab secret
1. Left sidebar key icon (Secrets) > Add new secret.
2. Name `HF_TOKEN`, value = the token from your `.env` (the account that was granted prov-gigapath access).
3. Switch **Notebook access** ON for this notebook.

## 4. Pick the GPU
Runtime > Change runtime type > **T4 GPU** > Save.

## 5. Run
Runtime > Run all. Expect:
- a Drive permission popup (allow);
- ~5-10 min setup (clone, pip, 4.5 GB model weights on the first extraction);
- then, for each class in `CLASSES` (default `['colon_n', 'colon_aca']`, in that order): zip hash check, unzip, extraction (157 batches of 32, roughly 10-25 min on a T4, progress is printed), checks, copy to Drive;
- per class the output prints `features: (5000, 1536) float32`, GPU/torch/timm versions and three sha256 lines, and the last cell prints `ALL CLASSES ON DRIVE`.

Re-running and options:
- **Disconnect or error:** just Run all again. A class whose folder `MyDrive/btp_repro/gigapath_out/<cls>/` already exists is skipped (its zip is not even needed). A class folder only appears on Drive after all three files are copied and verified, so a half-finished copy is never mistaken for a result (a leftover `<cls>.tmp` folder is replaced).
- **One class only:** edit `CLASSES` in cell 1, e.g. `['colon_aca']`.
- **CUDA out of memory:** lower `BATCH_SIZE` in cell 1 from 32 to 16, re-run cell 1, then cell 6 (everything before is cached in the runtime).
- A fresh runtime (after a long idle) repeats the setup cells; finished classes are still skipped.

## 6. Bring results back
1. In Drive, download `MyDrive/btp_repro/gigapath_out/colon_n/` and `MyDrive/btp_repro/gigapath_out/colon_aca/` (3 files each: `features.npy`, `ids.npy`, `ids_2_img_paths.json`).
2. Put each class's three files at
   `~/btp_dedup_uni/third_party/LC25000-clean/feature_vectors/<cls>/prov-gigapath/imagenet/`
   (create the directories; e.g. copy via `\\wsl$\...` or `cp`).
3. Paste the GPU/torch/timm line and the sha256 lines (six in total) from the notebook output to me. I will verify them against the local files, check that
   each `ids_2_img_paths.json` matches the other extractors' ordering for that class (e.g. the local UNI `resize_only` one), and then include prov-gigapath in the Phase 3 evaluation for both classes.

## Notes
- Upstream code is unmodified (pinned commit 912f342, checked again at the end of the notebook); fp32; no `--precision half`.
- GPU features will differ from a CPU run in the last float bits; any ARI/P@k difference from that is a reportable caveat.
- The notebook logic (skip, resume, hash/count/shape checks, atomic copy) was tested locally against a stub extractor and a temp folder in place of Drive. It has not been run on Colab itself.
