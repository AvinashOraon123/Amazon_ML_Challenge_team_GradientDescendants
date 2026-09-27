"""Full-scale local tuning: LightGBM vs XGBoost (the backend of the running job), bigger models, ensembles.

Uses the v2 feature table from the error-analysis job plus the new features computed locally (cached)."""
import sys, time, json, gc, os
sys.path.insert(0, '../src')
import numpy as np, polars as pl, lightgbm as lgb, xgboost as xgb
from ber import data, decide, normalize as N, features as F

t0 = time.time()
CACHE = 'cache_full'
os.makedirs(CACHE, exist_ok=True)
feat = pl.read_parquet('../../../../kaggle_staging/an2/analysis/artifacts/train/features.parquet')
cols = json.load(open('../../../../kaggle_staging/results_v2/models/matcher/feature_columns.json'))
recs = data.load_records('../../../dataset', 'train')
pairs = data.load_pairs('../../../dataset', recs.select('rid', 'entity_id'))
recs = data.add_holdout(recs)
src = recs['src'].to_numpy(); s1_all = np.nonzero(src == 1)[0]
s1_hold = s1_all[recs['holdout'].to_numpy()[s1_all]]
base = feat.select('s1_rid', 'rid')
y = feat['label'].to_numpy().astype(np.float32); fold = feat['fold'].to_numpy(); hold = feat['is_hold'].to_numpy()

if not os.path.exists(f'{CACHE}/X.npy'):
    need = pl.concat([feat.select(pl.col('s1_rid').alias('rid')), feat.select('rid')]).unique()
    sub = recs.join(need.with_columns(pl.col('rid').cast(pl.Int32)), on='rid')
    del recs; gc.collect()
    sub = N.normalize(sub)
    cc = ['rid', 'name_n', 'core_n', 'skel', 'addr_n', 'src', 'core_c', 'addr_c', 'legal']
    side = sub.select(cc); mk = sub.select('rid', *N.MARKERS)
    new = []
    for a in range(0, base.height, 1_000_000):
        c = base.slice(a, 1_000_000)
        p = (c.join(side.rename({k: f'{k}_1' for k in cc if k != 'rid'}), left_on='s1_rid', right_on='rid', how='left', maintain_order='left')
             .join(side.rename({k: f'{k}_2' for k in cc if k != 'rid'}), on='rid', how='left', maintain_order='left'))
        g = {**F.string_features(p), **F.rule_features(p), **F.token_features(p)}
        m = c.select('rid').join(mk, on='rid', how='left', maintain_order='left')
        for k in N.MARKERS:
            g[k] = m[k].to_numpy()
        new.append(pl.DataFrame(g)); del p
    new = pl.concat(new)
    add = [c for c in new.columns if c not in cols]
    X = np.hstack([feat.select(cols).to_numpy().astype(np.float32), new.select(add).to_numpy().astype(np.float32)])
    X = np.nan_to_num(X, nan=-9, posinf=9, neginf=-9)
    np.save(f'{CACHE}/X.npy', X)
    del new, sub, side, mk; gc.collect()
else:
    del recs; gc.collect()
X = np.load(f'{CACHE}/X.npy', mmap_mode='r')
print('X', X.shape, round(time.time() - t0), flush=True)


def f05(p):
    rule, res = decide.tune(base.with_columns(pl.Series('p', p)), pairs, s1_hold, log=lambda *_: None)
    return max(s for _, s in res)


def cv(fit, name):
    oof = np.zeros(len(X), np.float32)
    for k in range(5):
        tr, va = (fold != k) & hold, fold == k
        oof[va] = fit(np.asarray(X[tr]), y[tr], np.asarray(X[va & hold]), y[va & hold], np.asarray(X[va]))
    np.save(f'{CACHE}/oof_{name}.npy', oof)
    print(f'{name}: OOF F0.5 {f05(oof):.5f} ({time.time() - t0:.0f}s)', flush=True)
    return oof


def lgb_fit(leaves, lr, rounds):
    def fit(Xt, yt, Xv, yv, Xp):
        p = dict(objective='binary', learning_rate=lr, num_leaves=leaves, min_data_in_leaf=50, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
        b = lgb.train(p, lgb.Dataset(Xt, yt), rounds, valid_sets=[lgb.Dataset(Xv, yv)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        return b.predict(Xp, num_iteration=b.best_iteration)
    return fit


def xgb_fit(leaves, lr, rounds, depthwise=False):
    def fit(Xt, yt, Xv, yv, Xp):
        p = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', eta=lr,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, max_bin=256, min_child_weight=5, nthread=12)
        if depthwise:
            p.update(max_depth=10)
        else:
            p.update(max_depth=0, grow_policy='lossguide', max_leaves=leaves)
        dt = xgb.QuantileDMatrix(Xt, yt, max_bin=256)
        dv = xgb.QuantileDMatrix(Xv, yv, ref=dt)
        b = xgb.train(p, dt, rounds, evals=[(dv, 'v')], early_stopping_rounds=50, verbose_eval=False)
        return b.inplace_predict(Xp, iteration_range=(0, b.best_iteration + 1))
    return fit


runs = {}
runs['lgb_127_lr.06'] = cv(lgb_fit(127, 0.06, 1500), 'lgb_127_lr.06')          # current LightGBM setting
runs['xgb_127_lr.06'] = cv(xgb_fit(127, 0.06, 1500), 'xgb_127_lr.06')          # what the running job uses
runs['lgb_255_lr.04'] = cv(lgb_fit(255, 0.04, 3000), 'lgb_255_lr.04')
runs['xgb_depth10'] = cv(xgb_fit(0, 0.05, 2000, depthwise=True), 'xgb_depth10')
names = list(runs)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print(f'avg {names[i]} + {names[j]}: {f05((runs[names[i]] + runs[names[j]]) / 2):.5f}', flush=True)
print(f'avg all: {f05(sum(runs.values()) / len(runs)):.5f}', flush=True)
