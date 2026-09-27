"""Entity-level gate: P(entity has >= 1 match) from all its candidates, used to fix the empty/non-empty
decision (singleton false matches and wrongly empty entities = 25% of v2's F0.5 loss)."""
import sys, time
sys.path.insert(0, '../src')
import numpy as np, polars as pl, lightgbm as lgb
from ber import data, decide

t0 = time.time()
d = pl.read_parquet('../../../../kaggle_staging/an2/analysis/oof_pred.parquet').with_columns(
    pl.col('s1_rid').cast(pl.Int32), pl.col('rid').cast(pl.Int32))
recs = data.load_records('../../../dataset', 'train')
pairs = data.load_pairs('../../../dataset', recs.select('rid', 'entity_id')).with_columns(
    pl.col('s1_rid').cast(pl.Int32), pl.col('rid').cast(pl.Int32))
recs = data.add_holdout(recs)
src = recs['src'].to_numpy(); s1_all = np.nonzero(src == 1)[0]
s1_hold = s1_all[recs['holdout'].to_numpy()[s1_all]]
ev = pl.DataFrame({'s1_rid': s1_hold.astype(np.int32)})

# Source-1 side raw signals (country, lengths, noise markers of the entity's own record)
n, a = pl.col('name'), pl.col('addr')
s1 = recs.filter(pl.col('src') == 1).select(
    pl.col('rid').cast(pl.Int32).alias('s1_rid'),
    (pl.col('country') == 'India').cast(pl.Float32).alias('s1_india'),
    n.str.len_chars().cast(pl.Float32).alias('s1_name_len'), a.str.len_chars().cast(pl.Float32).alias('s1_addr_len'),
    n.str.split(' ').list.len().cast(pl.Float32).alias('s1_name_tok'),
    a.str.contains(r'[0-9]').cast(pl.Float32).alias('s1_addr_has_num'))
del recs

assigned = decide.assign(d.select('s1_rid', 'rid', 'p'))
best_of_rec = assigned.select('rid', pl.col('s1_rid').alias('rec_best_s1'), pl.col('p').alias('rec_best_p'))
x = d.join(best_of_rec, on='rid', how='left').with_columns(
    (pl.col('rec_best_s1') == pl.col('s1_rid')).alias('mine'))
x = x.sort(['s1_rid', 'p'], descending=[False, True])
ent = x.group_by('s1_rid').agg(
    pl.len().cast(pl.Float32).alias('n_cand'),
    pl.col('mine').sum().cast(pl.Float32).alias('n_assigned'),
    pl.col('p').max().alias('p_max'),
    pl.col('p').sum().alias('p_sum'),
    pl.col('p').filter(pl.col('mine')).max().fill_null(0).alias('pa1'),
    pl.col('p').filter(pl.col('mine')).sort(descending=True).get(1, null_on_oob=True).fill_null(0).alias('pa2'),
    pl.col('p').filter(pl.col('mine')).sum().alias('pa_sum'),
    (pl.col('p') > 0.5).sum().cast(pl.Float32).alias('n_gt05'),
    (pl.col('p') > 0.2).sum().cast(pl.Float32).alias('n_gt02'),
    ((1 - pl.col('p').filter(pl.col('mine')).clip(0, 0.999999)).log().sum()).exp().alias('p_none_assigned'),
    # the entity's best candidate: was it taken by another entity, and how did it look?
    pl.col('rec_best_p').first().alias('top_rec_best_p'),
    pl.col('mine').first().cast(pl.Float32).alias('top_is_mine'),
    pl.col('z_cos').first().alias('top_z_cos'), pl.col('zn_cos').first().alias('top_zn_cos'),
    pl.col('core_tset').first().alias('top_core_tset'), pl.col('addr_tset').first().alias('top_addr_tset'),
    pl.col('core_exact').first().alias('top_core_exact'), pl.col('addr_empty_2').first().alias('top_addr_empty'),
    pl.col('f_core_1').first().alias('f_core_1'), pl.col('num_first_eq').first().alias('top_num_eq'),
    pl.col('z_cos').max().alias('z_cos_max'), pl.col('addr_tset').max().alias('addr_tset_max'),
)
ent = ev.join(ent, on='s1_rid', how='left').join(s1, on='s1_rid', how='left').fill_null(0)
has = pairs.group_by('s1_rid').len('n_true')
ent = ent.join(has, on='s1_rid', how='left').with_columns(pl.col('n_true').fill_null(0))
y = (ent['n_true'].to_numpy() > 0).astype(np.float32)
fcols = [c for c in ent.columns if c not in ('s1_rid', 'n_true')]
X = ent.select(fcols).to_numpy().astype(np.float32)
fold = ((ent['s1_rid'].cast(pl.Int64) * 40503) % 1000 % 5).to_numpy()
print('entities', len(X), 'has-match rate', y.mean().round(4), round(time.time() - t0), 's', flush=True)

pne = np.zeros(len(X), np.float32)
for k in range(5):
    tr, va = fold != k, fold == k
    b = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
                       feature_fraction=0.9, verbose=-1), lgb.Dataset(X[tr], y[tr]), 600)
    pne[va] = b.predict(X[va])
imp = sorted(zip(b.feature_importance('gain'), fcols), reverse=True)[:10]
print('gate top features:', [(n_, int(g)) for g, n_ in imp], flush=True)
from sklearn.metrics import roc_auc_score
print('gate AUC', round(roc_auc_score(y, pne), 5), ' baseline AUC (p_none_assigned)', round(roc_auc_score(y, 1 - ent['p_none_assigned'].to_numpy()), 5), flush=True)

base_rule = {'rule': 'expected_f', 'alpha': 1.0, 'min_p': 0.0}
base_sel = decide.apply_rule(d.select('s1_rid', 'rid', 'p'), base_rule).select(pl.col('s1_rid').cast(pl.Int32), pl.col('rid').cast(pl.Int32))
f_base = decide.macro_f05(base_sel, pairs, s1_hold)
print(f'baseline (v2 decision): {f_base:.5f}', flush=True)
top1 = assigned.sort(['s1_rid', 'p'], descending=[False, True]).unique('s1_rid', keep='first', maintain_order=True).select(
    pl.col('s1_rid').cast(pl.Int32), pl.col('rid').cast(pl.Int32))
gate = ent.select('s1_rid').with_columns(pl.Series('pne', pne))
nonempty = base_sel.select('s1_rid').unique()
best = (f_base, None)
for lo in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
    for hi in (1.01, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7):
        kill = gate.filter(pl.col('pne') < lo).select('s1_rid')
        rescue = gate.filter(pl.col('pne') >= hi).join(nonempty, on='s1_rid', how='anti').select('s1_rid')
        sel = pl.concat([base_sel.join(kill, on='s1_rid', how='anti'), top1.join(rescue, on='s1_rid')])
        f = decide.macro_f05(sel, pairs, s1_hold)
        if f > best[0]:
            best = (f, (lo, hi))
        print(f'  kill<{lo:.2f} rescue>={hi:.2f}: {f:.5f}  (killed {kill.height}, rescued {rescue.height})', flush=True)
print(f'BEST gate: {best[0]:.5f} with {best[1]}  vs baseline {f_base:.5f}  (+{best[0] - f_base:.5f})', flush=True)
ent.with_columns(pl.Series('pne', pne)).write_parquet('cache_full/gate_entities.parquet')
