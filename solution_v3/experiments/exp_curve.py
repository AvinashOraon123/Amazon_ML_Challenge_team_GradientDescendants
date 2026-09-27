"""Learning curve for the stage-1 matcher at full scale: is more training data worth a K-fold encoder?"""
import sys, time, json
sys.path.insert(0, '../src')
import numpy as np, polars as pl, lightgbm as lgb
from ber import data, decide

t0 = time.time()
feat = pl.read_parquet('../../../../kaggle_staging/an2/analysis/artifacts/train/features.parquet')
cols = json.load(open('../../../../kaggle_staging/results_v2/models/matcher/feature_columns.json'))
recs = data.load_records('../../../dataset', 'train').select('rid', 'src', 'entity_id')
pairs = data.load_pairs('../../../dataset', recs)
recs = data.add_holdout(recs)
src = recs['src'].to_numpy()
s1_all = np.nonzero(src == 1)[0]
s1_hold = s1_all[recs['holdout'].to_numpy()[s1_all]]
del recs
X = feat.select(cols).to_numpy().astype(np.float32)
y = feat['label'].to_numpy().astype(np.float32)
fold = feat['fold'].to_numpy()
hold = feat['is_hold'].to_numpy()
s1 = feat['s1_rid'].to_numpy()
base = feat.select('s1_rid', 'rid')
print('loaded', X.shape, round(time.time() - t0), flush=True)
params = dict(objective='binary', learning_rate=0.1, num_leaves=127, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
for frac in (0.25, 0.5, 1.0):
    keep = ((s1.astype(np.int64) * 2246822519) % 1000) < frac * 1000     # subsample training ENTITIES
    oof = np.zeros(len(X), np.float32)
    for k in range(5):
        tr, va = (fold != k) & hold & keep, fold == k
        b = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=500)
        oof[va] = b.predict(X[va])
    rule, res = decide.tune(base.with_columns(pl.Series('p', oof)), pairs, s1_hold, log=lambda *_: None)
    print(f'train fraction {frac}: rows {int((hold & keep).sum()):,}  OOF F0.5 {max(s for _, s in res):.5f} ({time.time() - t0:.0f}s)', flush=True)
