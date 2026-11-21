"""Per-class paths for the repro scripts (--cls {colon_n,colon_aca}).

colon_n returns exactly the strings the scripts used before this module existed.
REPRO_OUT_ROOT (unset by default) redirects the repro/out tree; used only to re-run the
colon_n scripts against a scratch copy and compare the outputs byte for byte.
"""
import argparse
import os

ROOT = os.path.expanduser('~/btp_dedup_uni')
UP = f'{ROOT}/third_party/LC25000-clean'
CLASSES = ('colon_n', 'colon_aca')
# dimension of the UMAP recorded in their saved initial-clustering .pkl
THEIR_UMAP_DIM = {'colon_n': 8, 'colon_aca': 32}
N_FINAL_GROUPS = {'colon_n': 257, 'colon_aca': 249}  # len(final_clusters.json)


class Cls:
    def __init__(self, cls, umap_dim=None):
        assert cls in CLASSES, cls
        self.cls = cls
        self.their_dim = THEIR_UMAP_DIM[cls]
        self.umap_dim = umap_dim or self.their_dim
        self.k_final = N_FINAL_GROUPS[cls]
        out_root = os.environ.get('REPRO_OUT_ROOT', f'{ROOT}/repro/out')
        self.out = out_root if cls == 'colon_n' else f'{out_root}/{cls}'
        # initial clustering outputs: the primary run (UMAP dim = theirs) keeps the plain name
        self.initial = f'{self.out}/initial' if self.umap_dim == self.their_dim else f'{self.out}/initial_umap{self.umap_dim}'
        self.phase2 = f'{self.out}/phase2'
        self.phase3 = f'{self.out}/phase3'
        self.extras = f'{self.out}/phase3_extras'
        self.feat = f'{UP}/feature_vectors/{cls}/UNI/resize_only'
        self.ann = f'{UP}/annotations/{cls}/UNI/resize_only'
        self.pkl = f'{self.ann}/distance_metric=euclidean#dimensionality_reduction=UMAP-{self.their_dim}#clustering=kmeans.pkl'
        self.eval_glob = f'{UP}/eval_results/*{cls}*.csv'
        self.scratch_eval = f'{ROOT}/repro/scratch_eval' if cls == 'colon_n' else f'{ROOT}/repro/scratch_eval/{cls}'
        # my old scripts/stage_2-3 outputs
        self.emb = f'{ROOT}/embeddings' if cls == 'colon_n' else f'{ROOT}/embeddings/{cls}'
        self.res = f'{ROOT}/results' if cls == 'colon_n' else f'{ROOT}/results/{cls}'
        self.img_prefix = f'./LC25000/{cls}/'


def parse(description, umap_dim=False, extra=None):
    """Parse --cls (and optionally --umap-dim); returns (args, Cls)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cls', choices=CLASSES, default='colon_n', help='default: %(default)s')
    if umap_dim:
        parser.add_argument('--umap-dim', type=int, default=None,
                            help='UMAP dimension for my run; default = the dimension in their saved .pkl (colon_n 8, colon_aca 32)')
    if extra:
        extra(parser)
    args = parser.parse_args()
    return args, Cls(args.cls, getattr(args, 'umap_dim', None))
