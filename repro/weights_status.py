"""Load every extractor via upstream get_feature_extractor and run one forward pass on a random tensor (cpu)."""
import os, sys, time
sys.path.insert(0, os.path.expanduser('~/btp_dedup_uni/third_party/LC25000-clean'))
import torch
from source.feature_extraction.get_model import get_feature_extractor
names = ['imagenet_resnet18-last-layer','imagenet_resnet50-clam-extractor','dinov2_vits14','dinov2_vitb14',
         'simclr-tcga-lung_resnet18-2.5x','simclr-tcga-lung_resnet18-10x','owkin-phikon','UNI','prov-gigapath']
rows=[]
for n in names:
    t=time.time()
    try:
        m=get_feature_extractor(n).eval()
        with torch.no_grad(): o=m(torch.randn(2,3,224,224))
        o=o if torch.is_tensor(o) else o[0]
        rows.append((n,'OK',tuple(o.shape),f'{sum(p.numel() for p in m.parameters())/1e6:.0f}M params',f'{time.time()-t:.0f}s'))
        del m
    except Exception as e:
        rows.append((n,'FAIL',type(e).__name__+': '+str(e)[:100].replace('\n',' '),'',''))
    print(rows[-1],flush=True)
with open(os.path.expanduser('~/btp_dedup_uni/repro/weights_status.txt'),'w') as f:
    for r in rows: f.write('\t'.join(map(str,r))+'\n')
