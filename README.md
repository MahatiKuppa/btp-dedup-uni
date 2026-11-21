# btp-dedup-uni

Code for my BTP-1 (Chemical Engineering, IIT Madras): working out which of the 5,000 images in each LC25000 colon class (`colon_n`, `colon_aca`) are augmented copies of the same original, so the second phase can annotate one image per original and split the data without leakage.

The full write-up is in the BTP-1 report, "Deduplication of LC25000 Colon Histopathology Images for Microvascular Pattern Analysis".

## What's here

- `scripts/`: approach 3 (UNI and ResNet50 embeddings, then KMeans), matching to the published groups, the final representatives, the leakage check, re-scoring of approaches 1 and 2, and the diagnostic tests.
- `repro/`: the reproduction of Batchkala et al.'s LC25000-clean pipeline, plus the Colab notebook for Prov-GigaPath.
- `notebooks/`: the Colab notebooks for approaches 1 and 2.
- `results/`: the numbers quoted in the report.

## Running it

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env    # add a Hugging Face token with access to MahmoodLab/UNI
python scripts/stage_2_embed.py --cls colon_n    # or --cls colon_aca
```

The images aren't in the repo. They come from the original LC25000 release on [Academic Torrents](https://academictorrents.com/details/7a638ed187a6180fd6e464b3666a6ea0499af4af). The published groups come from [LC25000-clean](https://github.com/GeorgeBatch/LC25000-clean), commit `912f342` (see `repro/UPSTREAM_COMMIT.txt`).

Everything was run on a laptop CPU, apart from Prov-GigaPath, which needed a Colab GPU.
