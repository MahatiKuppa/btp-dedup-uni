# Patches to upstream (third_party/LC25000-clean @ 912f342)

Source edits: **none** so far. `extract_features.py --device cpu` takes the CPU path (model and inputs both on cpu).

Environment-only workarounds (outside the upstream tree, no code changed):
- torch.hub GitHub-API check returned 403 (`KeyError: 'Authorization'`). Fix: `git clone --depth 1 facebookresearch/dinov2` into `~/.cache/torch/hub/facebookresearch_dinov2_main` and add `facebookresearch` to `~/.cache/torch/hub/trusted_list`.
- SimCLR weights downloaded with gdown from dsmil-wsi Google Drive folder 1Rn_VpgM82VEfnjiVjDbObbBFHvs0V1OE into `weights/simclr-tcga-lung/` (gitignored upstream). Mapping is my inference (see weights_status.md).
