| extractor | status | out dim | note |
|---|---|---|---|
| imagenet_resnet18-last-layer | OK | 512 | torchvision IMAGENET1K_V1 |
| imagenet_resnet50-clam-extractor | OK | 1024 | torchvision resnet50 url |
| dinov2_vits14 | OK | 384 | torch.hub, see PATCHES.md |
| dinov2_vitb14 | OK | 768 | torch.hub |
| simclr-tcga-lung_resnet18-2.5x | OK | 512 | = dsmil `weights-low-mag/model-low-v1.pth` (INFERRED mapping) |
| simclr-tcga-lung_resnet18-10x | OK | 512 | = dsmil `model-v1.pth` (INFERRED mapping) |
| owkin-phikon | OK | 768 | |
| UNI | OK | 1024 | |
| prov-gigapath | **BLOCKED** | - | GatedRepoError 403: access not granted |

## prov-gigapath (access granted 2026-10-23)
HF access check passes. Downloaded tile-encoder files only (slide_encoder.pth and sample_data/ NOT downloaded), cache: `~/.cache/huggingface/hub/models--prov-gigapath--prov-gigapath/snapshots/64f9e26c15019f2d4f6d9113c6822f88bb16b01b/`

| file | bytes | sha256 |
|---|---|---|
| config.json | 829 | 6937eb16679f8de311d6edd824930c27a2b1a1029a24907be59ced33a908ae55 |
| pytorch_model.bin | 4540023137 | 877947214318afa9e011754b74bbc3894a1f480a253afc7bc8045b8321dedd63 |

pytorch_model.bin sha256 equals the LFS hash reported by the HF API.

### RAM probe (`repro/gigapath_ram_probe.py`, logs in `repro/logs/gigapath_ram_*.log`)
Machine: 7787 MB total, 2048 MB swap, ~6.1 GB MemAvailable at start (nothing else running).
Load via upstream `get_feature_extractor('prov-gigapath')` (fp32, `pytorch_model.bin` is not mmap-able):
- RSS 4.8 GB after reading the state dict (about the file size), then 5.4 GB, then 6.0 GB while building/copying into the model.
- MemAvailable fell to 359 MB and swap to 179 MB free, so my watchdog killed the probe (WATCHDOG_KILLED) **during model load, before any forward pass**. Without the watchdog the kernel OOM-killer or a WSL stall would have followed.
- Extrapolated fp32 load peak: ~9 GB (state dict 4.5 GB + model 4.5 GB, ~1.1B params). Steady state after load ~4.6 GB + activations. Forward numbers (bs=1) not measured.

### Colab input zip
`data/colon_n_original.zip` (gitignored): 5000 `colonn*.jpeg`, flat, original filenames, stored (no compression), 471608147 bytes.
sha256: `14724d148ccd08631c8977a91e4ef1d5871e959831d29997832ba15bc240169d`

### SimCLR 2.5x/10x mapping check (done)
Upstream `get_model.py` cites https://github.com/binli123/dsmil-wsi/issues/41. There the dsmil weight author-side comment (GeorgeBatch, "TCGA-lung weights", same Drive folder 1Rn_VpgM82VEfnjiVjDbObbBFHvs0V1OE) says: "magnification: low=2.5x, high=10x" and "pre-training: v0 for 3 days, v1 for 2 weeks (better results)".
So: `weights-low-mag/model-low-v1.pth` -> 2.5x (low); top-level `model-v1.pth` -> 10x (high); v1 = the 2-week run, matching upstream's `model-v1.pth` filename. Mapping is consistent, but the issue page does not name individual files; "top-level = high" is by elimination. Residual risk is low.
