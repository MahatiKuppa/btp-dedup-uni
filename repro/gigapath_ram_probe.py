"""Measure RAM of loading prov-gigapath via upstream get_feature_extractor + one batch-of-1 forward (cpu). No extraction."""
import os, sys, time, resource
sys.path.insert(0, os.path.expanduser('~/btp_dedup_uni/third_party/LC25000-clean'))
import torch
def rss(): return int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0]) / 1024
def hwm(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f'start rss={rss():.0f}MB', flush=True)
from source.feature_extraction.get_model import get_feature_extractor
t = time.time(); m = get_feature_extractor('prov-gigapath').eval()
print(f'loaded in {time.time()-t:.0f}s rss={rss():.0f}MB peak={hwm():.0f}MB params={sum(p.numel() for p in m.parameters())/1e6:.0f}M dtype={next(m.parameters()).dtype}', flush=True)
x = torch.randn(1, 3, 224, 224)
for i in range(2):
    t = time.time()
    with torch.no_grad(): o = m(x)
    print(f'forward bs=1 #{i}: {time.time()-t:.1f}s out={tuple(o.shape)} rss={rss():.0f}MB peak={hwm():.0f}MB', flush=True)
x = torch.randn(4, 3, 224, 224); t = time.time()
with torch.no_grad(): m(x)
print(f'forward bs=4: {time.time()-t:.1f}s rss={rss():.0f}MB peak={hwm():.0f}MB', flush=True)
