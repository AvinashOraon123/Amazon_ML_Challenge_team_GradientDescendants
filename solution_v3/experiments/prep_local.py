import sys, time; sys.path.insert(0, '.')
import polars as pl
from ber import data, normalize as N
t = time.time()
r = data.load_records('../../../dataset', 'train')
pairs = data.load_pairs('../../../dataset', r)
r = data.add_holdout(r)
r = N.normalize(r)
r.select('rid', 'src', 'country', 'holdout', 'name', 'addr', 'name_n', 'core_n', 'skel', 'addr_n',
         'core_c', 'addr_c', 'legal').write_parquet('../work_full/train/records.parquet')
pairs.write_parquet('../work_full/train/pairs.parquet')
print('done', r.height, round(time.time() - t), 's', flush=True)
