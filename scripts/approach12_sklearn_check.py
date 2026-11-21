#!/usr/bin/env python3
"""Does this sklearn reproduce Colab's KMeans on approach 2's saved colon_n features?

    <venv with the sklearn version to test>/bin/python -I scripts/approach12_sklearn_check.py

Fits KMeans(k=250, random_state=42, n_init=10) on data/approach12_drive/approach2/logs/colon_n_features.npy and
compares with colon_n_labels_baseline.npy. Result 2026-11-21 (numpy 2.5.5): 1.6.1 and 1.7.2 identical;
1.8.0 and 1.9.1 differ in 4717 labels (ARI 0.8238). See results/approach12_NOTES.md.
"""
import numpy as np, sklearn, sys, time
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
A='/home/mahati/btp_dedup_uni/data/approach12_drive/approach2/logs/'
f=np.load(A+'colon_n_features.npy'); b=np.load(A+'colon_n_labels_baseline.npy')
t=time.time(); l=KMeans(n_clusters=250,random_state=42,n_init=10).fit_predict(f)
print('sklearn',sklearn.__version__,'numpy',np.__version__,'identical',bool(np.array_equal(l,b)),'n_differ',int((l!=b).sum()),'ARI',round(float(adjusted_rand_score(b,l)),6),f'{time.time()-t:.0f}s',flush=True)
