"""Where is F0.5 lost? Per-entity decomposition of v2's out-of-fold predictions on held-out entities."""
import sys
sys.path.insert(0, '../src')
import numpy as np, polars as pl
from ber import data, decide

d = pl.read_parquet('../../../../kaggle_staging/an2/analysis/oof_pred.parquet')
recs = data.load_records('../../../dataset', 'train').select('rid', 'src', 'entity_id')
pairs = data.load_pairs('../../../dataset', recs).with_columns(pl.col('s1_rid').cast(pl.Int32), pl.col('rid').cast(pl.Int32))
recs = data.add_holdout(recs)
src = recs['src'].to_numpy(); s1_all = np.nonzero(src == 1)[0]
s1_hold = s1_all[recs['holdout'].to_numpy()[s1_all]]
ev = pl.DataFrame({'s1_rid': s1_hold.astype(np.int32)})

pred = decide.apply_rule(d.select('s1_rid', 'rid', 'p'), {'rule': 'expected_f', 'alpha': 1.0, 'min_p': 0.0}) \
    .select(pl.col('s1_rid').cast(pl.Int32), pl.col('rid').cast(pl.Int32))
tp = pairs.join(ev, on='s1_rid'); pp = pred.join(ev, on='s1_rid')
hit = pp.join(tp, on=['s1_rid', 'rid'], how='semi')
per = (ev.join(pp.group_by('s1_rid').len('n_pred'), on='s1_rid', how='left')
       .join(tp.group_by('s1_rid').len('n_true'), on='s1_rid', how='left')
       .join(hit.group_by('s1_rid').len('n_hit'), on='s1_rid', how='left').fill_null(0))
n = per.height
per = per.with_columns(
    (pl.col('n_hit') / pl.col('n_pred').clip(1, None)).alias('P'),
    (pl.col('n_hit') / pl.col('n_true').clip(1, None)).alias('R'))
per = per.with_columns(
    pl.when((pl.col('n_pred') == 0) & (pl.col('n_true') == 0)).then(1.0)
    .when((pl.col('P') + pl.col('R')) > 0).then(1.25 * pl.col('P') * pl.col('R') / (0.25 * pl.col('P') + pl.col('R')))
    .otherwise(0.0).alias('F'))
per = per.with_columns(
    pl.when(pl.col('n_true') == 0).then(pl.when(pl.col('n_pred') == 0).then(pl.lit('singleton: correct (empty)')).otherwise(pl.lit('singleton: FALSE MATCH (F=0)')))
    .when(pl.col('n_pred') == 0).then(pl.lit('has matches: predicted EMPTY (F=0)'))
    .when(pl.col('n_hit') == 0).then(pl.lit('has matches: all predictions wrong (F=0)'))
    .when((pl.col('n_hit') == pl.col('n_true')) & (pl.col('n_pred') == pl.col('n_hit'))).then(pl.lit('has matches: perfect'))
    .when(pl.col('n_pred') > pl.col('n_hit')).then(pl.lit('has matches: some wrong merges'))
    .otherwise(pl.lit('has matches: some missed')).alias('case'))
tot_loss = float((1 - per['F']).sum())
print(f'entities {n:,}  macro F0.5 {per["F"].mean():.5f}  total loss {tot_loss / n:.5f}')
out = per.group_by('case').agg(pl.len().alias('entities'), (1 - pl.col('F')).sum().alias('loss')).with_columns(
    (pl.col('entities') / n * 100).round(2).alias('% of entities'),
    (pl.col('loss') / n).round(5).alias('F0.5 lost'),
    (pl.col('loss') / tot_loss * 100).round(1).alias('% of loss')).sort('loss', descending=True)
pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(60)
print(out)
# singleton false matches: how confident were they?
sing = per.filter(pl.col('case') == 'singleton: FALSE MATCH (F=0)').select('s1_rid')
fp_sing = pp.join(sing, on='s1_rid').join(d.select(pl.col('s1_rid').cast(pl.Int32), pl.col('rid').cast(pl.Int32), 'p', 'addr_empty_2', 'core_exact'), on=['s1_rid', 'rid'])
print('false-match singletons: predicted pairs', fp_sing.height, ' p quantiles', [round(fp_sing['p'].quantile(q), 3) for q in (.1, .25, .5, .75, .9)],
      ' addr empty share', round(fp_sing['addr_empty_2'].mean(), 3))
emp = per.filter(pl.col('case') == 'has matches: predicted EMPTY (F=0)')
print('predicted-empty entities with matches: n_true distribution', emp['n_true'].value_counts().sort('n_true').rows()[:8])
per.write_parquet('cache_full/per_entity_v2.parquet')
