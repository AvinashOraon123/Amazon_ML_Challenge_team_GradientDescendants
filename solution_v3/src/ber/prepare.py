"""Stage 1: raw tables -> normalised record table + fixed-width byte matrices, saved under <work>/<split>/."""
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from . import data, normalize
from .encoder import EncoderConfig


def _padded(col):
    # leading/trailing space so word-boundary n-grams exist; empty stays empty (=> no n-grams)
    return pl.when(pl.col(col) == "").then(pl.lit("")).otherwise(" " + pl.col(col) + " ")


def prepare(data_dir, work_dir, split, sample_pct=None):
    t0 = time.time()
    out = Path(work_dir) / split
    out.mkdir(parents=True, exist_ok=True)
    recs = data.load_records(data_dir, split)
    pairs = data.load_pairs(data_dir, recs) if split == "train" else None
    if sample_pct:
        if pairs is None:   # test has no labels: plain id-hash sample of every source
            recs = recs.filter(data.id_num() % 100 < sample_pct).drop("rid").with_row_index("rid")
            recs = recs.with_columns(pl.col("rid").cast(pl.Int32))
        else:
            recs, pairs = data.subsample(recs, pairs, sample_pct)
    recs = data.add_holdout(recs)
    print(f"[prepare] {split}: {recs.height:,} records loaded ({time.time() - t0:.0f}s)", flush=True)

    recs = normalize.normalize(recs)
    print(f"[prepare] normalised ({time.time() - t0:.0f}s)", flush=True)

    cfg = EncoderConfig()
    for field, col, width in (("name", "name_n", cfg.name_len), ("skel", "skel", cfg.skel_len),
                              ("addr", "addr_n", cfg.addr_len)):
        mat = normalize.to_bytes(recs.select(_padded(col)).to_series(), width)
        np.save(out / f"bytes_{field}.npy", mat)
    recs.write_parquet(out / "records.parquet")
    if pairs is not None:
        pairs.write_parquet(out / "pairs.parquet")
    stats = {
        "records": recs.height,
        "by_src": recs.group_by("src").len().sort("src").rows(),
        "by_country": recs.group_by("country").len().sort("country").rows(),
        "pairs": None if pairs is None else pairs.height,
        "seconds": round(time.time() - t0),
    }
    (out / "prepare_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[prepare] done: {stats}", flush=True)
