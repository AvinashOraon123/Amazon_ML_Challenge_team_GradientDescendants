"""Score every cached OOF prediction vector (and pairwise averages) with the challenge metric."""
import sys, glob, os, itertools
sys.path.insert(0, '../src')
import numpy as np, polars as pl
from ber import data, decide

feat = pl.read_parquet('../../../../kaggle_staging/an2/analysis/artifacts/train/features.parquet', columns=['s1_rid', 'rid'])
recs = data.load_records('../../../dataset', 'train').select('rid', 'src', 'entity_id')
pairs = data.load_pairs('../../../dataset', recs)
recs = data.add_holdout(recs)
src = recs['src'].to_numpy(); s1_all = np.nonzero(src == 1)[0]
s1_hold = s1_all[recs['holdout'].to_numpy()[s1_all]]
del recs


def f05(p):
    rule, res = decide.tune(feat.with_columns(pl.Series('p', p)), pairs, s1_hold, log=lambda *_: None)
    s = max(res, key=lambda r: r[1])
    return s[1], s[0]


oofs = {os.path.basename(f)[4:-4]: np.load(f) for f in sorted(glob.glob('cache_full/oof_*.npy'))}
for k, v in oofs.items():
    s, r = f05(v); print(f'{k:22s} {s:.5f}  {r}', flush=True)
for a, b in itertools.combinations(oofs, 2):
    s, r = f05((oofs[a] + oofs[b]) / 2); print(f'avg {a} + {b}: {s:.5f}', flush=True)
if len(oofs) > 2:
    s, r = f05(sum(oofs.values()) / len(oofs)); print(f'avg all {len(oofs)}: {s:.5f}', flush=True)
