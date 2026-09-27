"""Full-scale local test: v2 features vs v2 + (cleaned text, legal form, noise marker) features."""
import sys, time, json, gc
sys.path.insert(0, '../src')
import numpy as np, polars as pl, lightgbm as lgb
from ber import data, decide, normalize as N, features as F

t0 = time.time()
feat = pl.read_parquet('../../../../kaggle_staging/an2/analysis/artifacts/train/features.parquet')
cols = json.load(open('../../../../kaggle_staging/results_v2/models/matcher/feature_columns.json'))
recs = data.load_records('../../../dataset', 'train')
pairs = data.load_pairs('../../../dataset', recs.select('rid', 'entity_id'))
recs = data.add_holdout(recs)
src = recs['src'].to_numpy(); s1_all = np.nonzero(src == 1)[0]
s1_hold = s1_all[recs['holdout'].to_numpy()[s1_all]]
need = pl.concat([feat.select(pl.col('s1_rid').alias('rid')), feat.select('rid')]).unique()
sub = recs.join(need.with_columns(pl.col('rid').cast(pl.Int32)), on='rid')
del recs; gc.collect()
sub = N.normalize(sub)
print('normalised', sub.height, round(time.time() - t0), flush=True)

cc = ['rid', 'name_n', 'core_n', 'skel', 'addr_n', 'src', 'core_c', 'addr_c', 'legal']
side = sub.select(cc)
mk = sub.select('rid', *N.MARKERS)
new = []
base = feat.select('s1_rid', 'rid')
for a in range(0, base.height, 1_000_000):
    c = base.slice(a, 1_000_000)
    p = (c.join(side.rename({k: f'{k}_1' for k in cc if k != 'rid'}), left_on='s1_rid', right_on='rid', how='left', maintain_order='left')
         .join(side.rename({k: f'{k}_2' for k in cc if k != 'rid'}), on='rid', how='left', maintain_order='left'))
    g = {**F.string_features(p), **F.rule_features(p), **F.token_features(p)}
    m = c.select('rid').join(mk, on='rid', how='left', maintain_order='left')
    for k in N.MARKERS:
        g[k] = m[k].to_numpy()
    new.append(pl.DataFrame(g))
new = pl.concat(new)
add = [c for c in new.columns if c not in cols]
print('new feature columns:', add, round(time.time() - t0), flush=True)

y = feat['label'].to_numpy().astype(np.float32); fold = feat['fold'].to_numpy(); hold = feat['is_hold'].to_numpy()
params = dict(objective='binary', learning_rate=0.1, num_leaves=127, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
for name, X in [('v2 features', feat.select(cols).to_numpy().astype(np.float32)),
                ('v2 + new', np.hstack([feat.select(cols).to_numpy().astype(np.float32), new.select(add).to_numpy().astype(np.float32)]))]:
    oof = np.zeros(len(X), np.float32)
    for k in range(5):
        tr, va = (fold != k) & hold, fold == k
        b = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=500)
        oof[va] = b.predict(X[va])
    rule, res = decide.tune(base.with_columns(pl.Series('p', oof)), pairs, s1_hold, log=lambda *_: None)
    print(f'{name}: OOF F0.5 {max(s for _, s in res):.5f} ({time.time() - t0:.0f}s)', flush=True)
    if name == 'v2 + new':
        names = cols + add
        imp = sorted(zip(b.feature_importance('gain'), names), reverse=True)
        print('  new-feature gain ranks:', [(n, i) for i, (g, n) in enumerate(imp) if n in add][:15], flush=True)
    del X; gc.collect()
