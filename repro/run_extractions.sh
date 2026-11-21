#!/bin/bash
# Phase 3 extractions on CPU (usage: run_extractions.sh [colon_n|colon_aca]), cheapest first, skip combos whose features.npy exists. prov-gigapath is done on Colab.
CLS=${1:-colon_n}
cd ~/btp_dedup_uni/third_party/LC25000-clean
export HF_TOKEN=$(grep ^HF_TOKEN ~/btp_dedup_uni/.env | cut -d= -f2-)
PY=~/btp_dedup_uni/venv_repro/bin/python
LOG=~/btp_dedup_uni/repro/logs
[ "$CLS" = colon_n ] || { LOG=$LOG/$CLS; mkdir -p $LOG; }
COMBOS="imagenet_resnet18-last-layer:resize_only
imagenet_resnet18-last-layer:imagenet
simclr-tcga-lung_resnet18-2.5x:imagenet
simclr-tcga-lung_resnet18-10x:imagenet
imagenet_resnet50-clam-extractor:imagenet
dinov2_vits14:imagenet
owkin-phikon:resize_only
owkin-phikon:imagenet
dinov2_vitb14:imagenet
UNI:imagenet"
for c in $COMBOS; do
  m=${c%%:*}; n=${c##*:}
  if [ -f feature_vectors/$CLS/$m/$n/features.npy ]; then echo "$(date +%T) skip $m $n"; continue; fi
  echo "$(date +%T) start $m $n"
  $PY extract_features.py --cancer_type $CLS --img_norm $n --extractor_name $m --device cpu --batch_size 16 < /dev/null > $LOG/extract_${m}_${n}.log 2>&1 \
    && echo "$(date +%T) done $m $n" || echo "$(date +%T) FAILED $m $n"
done
echo "$(date +%T) ALL DONE"
