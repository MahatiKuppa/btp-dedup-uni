#!/bin/bash
# retrying installs into venv_repro
cd ~/btp_dedup_uni
export UV_HTTP_TIMEOUT=180
retry() { for i in 1 2 3 4 5; do "$@" && return 0; echo "retry $i"; sleep 5; done; return 1; }
retry uv pip install --python venv_repro/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu || exit 1
retry uv pip install --python venv_repro/bin/python "timm>=1.0.3" scikit-learn scipy statsmodels umap-learn transformers pandas numpy matplotlib seaborn tqdm huggingface_hub opencv-python-headless imageio scikit-image pytest nbformat
