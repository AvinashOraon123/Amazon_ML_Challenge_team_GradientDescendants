"""Full-scale local experiment: stage-3 stacker with per-source assignment features on v2 OOF predictions."""
import sys, time, json
sys.path.insert(0, '../src')
import numpy as np, polars as pl, lightgbm as lgb
from ber import data, decide

t0 = time.time()
d = pl.read_parquet('../../../../kaggle_staging/an2/analysis/oof_pred.parquet')
recs = data.load_records('../../../dataset', 'train').select('rid', 'src', 'entity_id')
pairs = data.load_pairs('../../../dataset', recs)
recs = data.add_holdout(recs)
src = recs['src'].to_numpy()
s1_all = np.nonzero(src == 1)[0]
s1_hold = s1_all[recs['holdout'].to_numpy()[s1_all]]
del recs
print('loaded', d.height, round(time.time() - t0), flush=True)

S = pl.Series(src)
d = d.with_columns(pl.col('rid').map_batches(lambda r: pl.Series(src[r.to_numpy()])).alias('rsrc'))


def structural(d, pcol='p'):
    """Per entity: confident records by source; per record: how its best entity compares."""
    x = d.with_columns((pl.col(pcol) > 0.5).cast(pl.Int32).alias('conf'))
    x = x.with_columns(
        (pl.col('conf') * (pl.col('rsrc') == 2)).sum().over('s1_rid').alias('s1_c2'),
        (pl.col('conf') * (pl.col('rsrc') == 3)).sum().over('s1_rid').alias('s1_c3'),
        (pl.col(pcol) * (pl.col('rsrc') == 2)).sum().over('s1_rid').alias('s1_p2'),
        (pl.col(pcol) * (pl.col('rsrc') == 3)).sum().over('s1_rid').alias('s1_p3'),
    )
    x = x.with_columns(
        # confident records of the SAME source as this record, excluding itself
        (pl.when(pl.col('rsrc') == 2).then(pl.col('s1_c2')).otherwise(pl.col('s1_c3')) - pl.col('conf')).alias('same_src_conf'),
        (pl.when(pl.col('rsrc') == 2).then(pl.col('s1_c3')).otherwise(pl.col('s1_c2'))).alias('other_src_conf'),
        (pl.when(pl.col('rsrc') == 2).then(pl.col('s1_p2')).otherwise(pl.col('s1_p3')) - pl.col(pcol)).alias('same_src_psum'),
        (pl.when(pl.col('rsrc') == 2).then(pl.col('s1_p3')).otherwise(pl.col('s1_p2'))).alias('other_src_psum'),
        pl.col(pcol).rank('ordinal', descending=True).over('rid').alias('rec_rank'),
        pl.col(pcol).max().over('rid').alias('rec_max'),
        pl.col(pcol).sum().over('rid').alias('rec_sum'),
        pl.len().over('rid').alias('rec_n'),
    )
    # among the record's exact-name candidates: is this the one lacking a same-source record?
    x = x.with_columns(
        (pl.col('same_src_conf') == 0).cast(pl.Int32).alias('needs_src'),
    ).with_columns(
        (pl.col('needs_src') * pl.col('core_exact')).sum().over('rid').alias('rec_n_needy_exact'),
    )
    return x


feat_cols = ['p', 'core_exact', 'f_core_1', 'f_core_2', 'addr_empty_2', 'rec_n_core_exact', 's1_n_core_exact',
             'z_cos', 'zn_cos', 'lead_core_tset', 'lead_z_cos', 'rsrc', 's1_ncand', 'rec_ncand',
             'same_src_conf', 'other_src_conf', 'same_src_psum', 'other_src_psum', 'rec_rank', 'rec_max', 'rec_sum',
             'rec_n', 'needs_src', 'rec_n_needy_exact']

fold = ((d['rid'].cast(pl.Int64) * 2654435761) % 1000 % 5).to_numpy()
hold = d['is_hold'].to_numpy()
y = d['label'].to_numpy().astype(np.float32)
base = d.select('s1_rid', 'rid')

print('baseline (v2 p):', decide.tune(base.with_columns(d['p']), pairs, s1_hold, log=lambda *_: None)[0],
      round(max(s for _, s in decide.tune(base.with_columns(d['p']), pairs, s1_hold, log=lambda *_: None)[1]), 5), flush=True)

p_cur = d['p'].to_numpy()
for it in range(2):   # iterate: structural features from the previous round's probabilities
    x = structural(d.with_columns(pl.Series('p', p_cur)))
    X = x.select(feat_cols).to_numpy().astype(np.float32)
    oof = np.zeros(len(X), np.float32)
    for k in range(5):
        tr, va = (fold != k) & hold, fold == k
        b = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                           feature_fraction=0.9, verbose=-1), lgb.Dataset(X[tr], y[tr]), num_boost_round=400)
        oof[va] = b.predict(X[va])
    rule, res = decide.tune(base.with_columns(pl.Series('p', oof)), pairs, s1_hold, log=lambda *_: None)
    print(f'stage-3 round {it}: best F0.5 {max(s for _, s in res):.5f} rule {rule} ({time.time() - t0:.0f}s)', flush=True)
    imp = sorted(zip(b.feature_importance('gain'), feat_cols), reverse=True)[:10]
    print('  top features:', [(n, int(g)) for g, n in imp], flush=True)
    p_cur = oof
