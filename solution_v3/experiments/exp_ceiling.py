"""Realistic ceiling: perfect matcher except for pairs that are unreachable or genuinely ambiguous."""
import sys
sys.path.insert(0, '../src')
import numpy as np, polars as pl
from ber import data, decide

d = pl.read_parquet('../../../../kaggle_staging/an2/analysis/oof_pred.parquet')
recs = data.load_records('../../../dataset', 'train').select('rid', 'src', 'entity_id')
pairs = data.load_pairs('../../../dataset', recs).with_columns(pl.col('s1_rid').cast(pl.Int32), pl.col('rid').cast(pl.Int32))
recs = data.add_holdout(recs)
src = recs['src'].to_numpy()
s1_all = np.nonzero(src == 1)[0]
s1_hold = s1_all[recs['holdout'].to_numpy()[s1_all]]
hold = pl.DataFrame({'s1_rid': s1_hold.astype(np.int32)})
tp = pairs.join(hold, on='s1_rid')
cand = d.select(pl.col('s1_rid').cast(pl.Int32), pl.col('rid').cast(pl.Int32), 'addr_empty_2', 'core_exact', 'f_core_2', 'p')
tpc = tp.join(cand, on=['s1_rid', 'rid'], how='left')
in_cand = tpc['p'].is_not_null()
ambig = in_cand & (tpc['addr_empty_2'] == 1) & (tpc['f_core_2'] >= 2)
print('held-out true pairs', tp.height, '| not in candidates', int((~in_cand).sum()), '| ambiguous shared-name no-address', int(ambig.sum()))
cur = decide.apply_rule(d.select('s1_rid', 'rid', 'p'), {'rule': 'expected_f', 'alpha': 1.0, 'min_p': 0.0})
print('current v2 F0.5:', round(decide.macro_f05(cur, pairs, s1_hold), 5))
for name, keep in [('perfect matcher on all candidates', in_cand),
                   ('perfect, but ambiguous no-address pairs lost', in_cand & ~ambig)]:
    pred = tpc.filter(keep).select('s1_rid', 'rid')
    print(f'{name}: {decide.macro_f05(pred, pairs, s1_hold):.5f}')
# how many of the ambiguous ones does v2 already get right?
got = tpc.filter(ambig).join(cur.select(pl.col('s1_rid').cast(pl.Int32), pl.col('rid').cast(pl.Int32)), on=['s1_rid', 'rid'], how='semi')
print('ambiguous pairs v2 already predicts correctly:', got.height)
