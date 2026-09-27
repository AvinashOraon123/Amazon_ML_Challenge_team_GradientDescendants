"""Specialist for entities with exactly ONE strong candidate: true 1-match entity vs singleton + decoy copy.
The global model averages these in with everything else; a model trained only on them may separate better."""
import sys, time, json
sys.path.insert(0, '../src')
import numpy as np, polars as pl, lightgbm as lgb
from ber import data, decide

t0 = time.time()
feat = pl.read_parquet('../../../../kaggle_staging/an2/analysis/artifacts/train/features.parquet',
                       columns=['s1_rid', 'rid', 'label', 'is_hold', 'fold'])
X = np.load('cache_full/X.npy', mmap_mode='r')
p = (np.load('cache_full/oof_lgb_127_lr.06.npy') + np.load('cache_full/oof_xgb_127_lr.06.npy')) / 2
recs = data.load_records('../../../dataset', 'train').select('rid', 'src', 'entity_id')
pairs = data.load_pairs('../../../dataset', recs)
recs = data.add_holdout(recs)
src = recs['src'].to_numpy(); s1_all = np.nonzero(src == 1)[0]
s1_hold = s1_all[recs['holdout'].to_numpy()[s1_all]]
del recs
base = feat.select('s1_rid', 'rid').with_columns(pl.Series('p', p))


def score(pv):
    rule, res = decide.tune(base.with_columns(pl.Series('p', pv)), pairs, s1_hold, log=lambda *_: None)
    return max(s for _, s in res)


f0 = score(p)
print(f'baseline (lgb+xgb avg): {f0:.5f}', flush=True)

# entity context from the baseline probabilities
d = feat.with_columns(pl.Series('p', p), pl.int_range(pl.len()).alias('row'))
asg = decide.assign(d.select('s1_rid', 'rid', 'p'))
mine = asg.select('s1_rid', 'rid').with_columns(pl.lit(True).alias('mine'))
d = d.join(mine, on=['s1_rid', 'rid'], how='left').with_columns(pl.col('mine').fill_null(False))
ent = d.filter(pl.col('mine')).group_by('s1_rid').agg(
    (pl.col('p') > 0.3).sum().alias('n_strong'), pl.col('p').max().alias('pmax'),
    pl.col('p').sort(descending=True).get(1, null_on_oob=True).fill_null(0).alias('p2'))
d = d.join(ent, on='s1_rid', how='left').with_columns(pl.col('n_strong').fill_null(0), pl.col('p2').fill_null(0))
# the group: this pair is its entity's single strong assigned candidate
grp = d.filter(pl.col('mine') & (pl.col('n_strong') == 1) & (pl.col('p') > 0.3))
rows = grp['row'].to_numpy()
print('group pairs', len(rows), ' true-match rate', round(float(grp['label'].mean()), 4),
      ' held-out', int(grp['is_hold'].sum()), flush=True)
Xg = np.hstack([np.asarray(X[rows]), grp.select('p', 'p2', 'pmax').to_numpy().astype(np.float32),
                (src[grp['rid'].to_numpy()] == 3).astype(np.float32)[:, None]])
yg = grp['label'].to_numpy().astype(np.float32)
fg = grp['fold'].to_numpy(); hg = grp['is_hold'].to_numpy()
spec = np.zeros(len(rows), np.float32)
for k in range(5):
    tr, va = (fg != k) & hg, fg == k
    b = lgb.train(dict(objective='binary', learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.7,
                       bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1), lgb.Dataset(Xg[tr], yg[tr]), 600)
    spec[va] = b.predict(Xg[va])
from sklearn.metrics import roc_auc_score
m = hg
print('group AUC: global model', round(roc_auc_score(yg[m], grp['p'].to_numpy()[m]), 5), ' specialist', round(roc_auc_score(yg[m], spec[m]), 5), flush=True)
for w in (0.25, 0.5, 0.75, 1.0):
    pv = p.copy()
    pv[rows] = (1 - w) * p[rows] + w * spec
    print(f'  specialist weight {w}: F0.5 {score(pv):.5f}  (baseline {f0:.5f})', flush=True)
print(f'done {time.time() - t0:.0f}s', flush=True)
