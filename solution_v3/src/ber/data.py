"""Loading the raw TSVs into one record table, plus ground-truth pairs and the validation split.

Record table columns:
    rid        int32  row id, dense 0..N-1 over S1, S2, S3 (in that order)
    entity_id  str
    src        int8   1, 2 or 3
    country    str    open set of labels (never filtered or hard-coded)
    name, addr str    raw business_name / business_address ("" when missing)
"""
from pathlib import Path

import polars as pl

import os

HOLDOUT_MOD = 10      # 1 in 10 Source-1 entities is held out for validation by default
HOLDOUT_REM = 7


def holdout_digits():
    """Hundreds digits of held-out Source-1 ids. BER_HOLDOUT_PCT=50 holds out 5 digits (7 and 3,4,5,6):
    the encoder trains on the rest, the matcher trains on (5x more) entities the encoder never saw."""
    k = max(1, int(os.environ.get("BER_HOLDOUT_PCT", "10")) // 10)
    return [HOLDOUT_REM] + [d for d in (3, 4, 5, 6, 8, 2, 1, 9, 0) if d != HOLDOUT_REM][: k - 1]


def read_table(data_dir, split, stem):
    """Read `stem` as the original TSV (dataset/<split>/<stem>.tsv) or a flat parquet copy (<data_dir>/<stem>.parquet).

    The parquet copy is a lossless conversion used for the Kaggle upload (smaller, faster to read).
    """
    data_dir = Path(data_dir)
    for p in (data_dir / f"{stem}.parquet", data_dir / split / f"{stem}.parquet"):
        if p.exists():
            return pl.read_parquet(p).fill_null("")
    # quote_char=None: fields are never quoted, and names may contain '"' characters
    df = pl.read_csv(data_dir / split / f"{stem}.tsv", separator="\t", quote_char=None, infer_schema=False)
    return df.fill_null("")


def id_num(col="entity_id"):
    """Numeric part of an entity id ('S2-12345' -> 12345), used for stable hashing."""
    return pl.col(col).str.slice(3).cast(pl.Int64)


def load_records(data_dir, split):
    data_dir = Path(data_dir)
    parts = []
    for s in (1, 2, 3):
        df = read_table(data_dir, split, f"{split}_source{s}")
        parts.append(df.select(
            pl.col("entity_id"),
            pl.lit(s, dtype=pl.Int8).alias("src"),
            pl.col("country"),
            pl.col("business_name").alias("name"),
            pl.col("business_address").alias("addr"),
        ))
    recs = pl.concat(parts)
    return recs.with_row_index("rid").with_columns(pl.col("rid").cast(pl.Int32))


def load_pairs(data_dir, recs):
    """Ground-truth match pairs as (s1_rid, rid)."""
    gt = read_table(data_dir, "train", "train_ground_truth")
    ids = recs.select("entity_id", "rid")
    pairs = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
        .join(ids.rename({"entity_id": "source1_entity_id", "rid": "s1_rid"}), on="source1_entity_id")
        .join(ids.rename({"entity_id": "matched_entity_ids"}), on="matched_entity_ids")
        .select("s1_rid", "rid")
    )
    return pairs


def add_holdout(recs):
    """Mark held-out Source-1 entities (10% by default, deterministic on the id)."""
    return recs.with_columns(
        # uses the hundreds digit, independent of `subsample` (last two digits)
        ((pl.col("src") == 1) & ((id_num() // 100) % HOLDOUT_MOD).is_in(holdout_digits())).alias("holdout")
    )


def subsample(recs, pairs, pct):
    """Keep pct% of Source-1 entities, all their matched records, and pct% of the other records.

    Keeps the matched/unmatched ratio of the full data, so blocking statistics stay representative.
    """
    keep_s1 = recs.filter((pl.col("src") == 1) & (id_num() % 100 < pct)).select("rid")
    kept_pairs = pairs.join(keep_s1.rename({"rid": "s1_rid"}), on="s1_rid")
    matched_any = pairs.select("rid")
    others = recs.filter(pl.col("src") != 1).join(matched_any, on="rid", how="anti")
    others = others.filter(id_num() % 100 < pct).select("rid")
    keep = pl.concat([keep_s1, kept_pairs.select("rid"), others]).unique()
    sub = recs.join(keep, on="rid").sort("rid")
    # re-index densely
    remap = sub.select(pl.col("rid").alias("old"), pl.int_range(pl.len(), dtype=pl.Int32).alias("rid"))
    sub = sub.drop("rid").with_columns(remap["rid"]).select(recs.columns)
    kept_pairs = (
        kept_pairs.join(remap.rename({"old": "s1_rid", "rid": "n1"}), on="s1_rid")
        .join(remap.rename({"old": "rid", "rid": "n2"}), on="rid")
        .select(pl.col("n1").alias("s1_rid"), pl.col("n2").alias("rid"))
    )
    return sub, kept_pairs
