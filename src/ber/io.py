"""Data access: raw TSVs from S3 (cached locally) and dual local + S3 checkpoints.

Design choice: boto3 rather than s3fs / pandas "s3://" paths.
  * explicit control over what is transferred and when (no silent re-reads);
  * one dependency with stable semantics for upload/download/head_object;
  * a local cache means each raw file crosses the network at most once per instance.

Cost notes (us-east-1 list prices; check your region):
  * S3 -> EC2 in the SAME region: data transfer is free; you pay only GET requests
    ($0.0004 per 1k) - negligible. Cross-region reads cost ~$0.02/GB, so keep the
    bucket and instance in the same region.
  * Storage: ~2.5 GB raw + ~10-20 GB of checkpoints ~= $0.25-0.50 / month (S3 Standard).
  * The local cache is mainly about speed: re-reading a 500 MB TSV from S3 every
    notebook run costs ~5-10 s of wall time vs. <1 s for local Parquet.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from . import config as C

_s3_client = None

# The raw files have no quoting and contain stray quote characters in names, so
# QUOTE_NONE (3) is required; keep_default_na=False keeps "NULL"/"NA" as text and
# empty cells as "" rather than NaN.
TSV_READ_KW = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3, encoding="utf-8")


def _s3():
    global _s3_client
    if _s3_client is None:
        import boto3

        _s3_client = boto3.client("s3")
    return _s3_client


def s3_key(*parts: str) -> str:
    """Join key parts, tolerating an empty prefix (bucket root)."""
    return "/".join(p.strip("/") for p in parts if p and p.strip("/"))


def _ckpt_key(rel: str) -> str:
    return s3_key(C.S3_PREFIX, C.CHECKPOINT_S3_SUBDIR, rel)


def _s3_exists(key: str) -> bool:
    if not C.s3_enabled():
        return False
    from botocore.exceptions import ClientError

    try:
        _s3().head_object(Bucket=C.S3_BUCKET, Key=key)
        return True
    except ClientError:
        return False


def s3_upload(local: Path, key: str, force: bool = False) -> None:
    """Mirror a file to S3. Files above C.S3_SYNC_MAX_MB are kept local-only unless
    force=True: the S3 free tier is 5 GB / ~2,000 PUTs per month, and the EBS volume already
    survives instance stop/start, so only small, valuable artifacts are mirrored."""
    if not C.s3_enabled():
        return
    mb = local.stat().st_size / 1e6
    if not force and C.S3_SYNC_MAX_MB is not None and mb > C.S3_SYNC_MAX_MB:
        print(f"  [s3] skipped {local.name} ({mb:,.0f} MB > S3_SYNC_MAX_MB={C.S3_SYNC_MAX_MB}; kept on local disk)")
        return
    t = time.time()
    _s3().upload_file(str(local), C.S3_BUCKET, key)
    print(f"  [s3] uploaded {mb:,.1f} MB -> s3://{C.S3_BUCKET}/{key} ({time.time() - t:.1f}s)")


def s3_download(key: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    _s3().download_file(C.S3_BUCKET, key, str(local))
    mb = local.stat().st_size / 1e6
    print(f"  [s3] downloaded s3://{C.S3_BUCKET}/{key} ({mb:,.1f} MB, {time.time() - t:.1f}s)")


# --------------------------------------------------------------------------------------
# Raw data
# --------------------------------------------------------------------------------------
RAW_FILES = {
    "s1": "{split}_source1.tsv",
    "s2": "{split}_source2.tsv",
    "s3": "{split}_source3.tsv",
    "gt": "{split}_ground_truth.tsv",
}


def raw_path(split: str, name: str) -> Path:
    """Local path of a raw TSV, downloading it from S3 on first use.

    Lookup order: BER_RAW_DIR (pre-existing local copy) -> local cache -> S3.
    """
    fname = RAW_FILES[name].format(split=split)
    if C.RAW_LOCAL_DIR is not None:
        p = C.RAW_LOCAL_DIR / split / fname
        if p.exists():
            return p
    p = C.RAW_CACHE_DIR / split / fname
    if p.exists():
        return p
    if not C.s3_enabled():
        raise FileNotFoundError(
            f"{fname} not found locally and S3 is not configured. Set BER_RAW_DIR to the "
            f"folder containing train/ and test/, or set BER_S3_BUCKET."
        )
    s3_download(s3_key(C.S3_PREFIX, C.RAW_S3_SUBDIR, split, fname), p)
    return p


def read_raw(split: str, name: str, usecols=None) -> pd.DataFrame:
    t = time.time()
    df = pd.read_csv(raw_path(split, name), usecols=usecols, **TSV_READ_KW)
    print(f"  read {split}/{name}: {len(df):,} rows in {time.time() - t:.1f}s")
    return df


# --------------------------------------------------------------------------------------
# Checkpoints (Parquet / arbitrary files), mirrored local <-> S3
# --------------------------------------------------------------------------------------
def ckpt_path(rel: str) -> Path:
    return C.CHECKPOINT_DIR / rel


def exists(rel: str) -> bool:
    """True when a checkpoint is available locally or on S3."""
    return ckpt_path(rel).exists() or _s3_exists(_ckpt_key(rel))


def save_parquet(df: pd.DataFrame, rel: str, sync: bool = True) -> Path:
    p = ckpt_path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(p)  # atomic: a crash never leaves a half-written checkpoint
    print(f"  saved {rel}: {len(df):,} rows, {p.stat().st_size / 1e6:,.1f} MB")
    if sync:
        s3_upload(p, _ckpt_key(rel))
    return p


def load_parquet(rel: str, columns=None) -> pd.DataFrame:
    p = fetch(rel)
    return pd.read_parquet(p, columns=columns)


def save_file(local: Path, rel: str | None = None, force: bool = False) -> None:
    """Mirror an arbitrary artifact (model, json, npz) that already exists in CHECKPOINT_DIR."""
    rel = rel or str(local.relative_to(C.CHECKPOINT_DIR)).replace("\\", "/")
    s3_upload(local, _ckpt_key(rel), force=force)


def fetch(rel: str) -> Path:
    """Local path of a checkpoint, pulling it from S3 if the local copy is missing."""
    p = ckpt_path(rel)
    if not p.exists():
        if not _s3_exists(_ckpt_key(rel)):
            raise FileNotFoundError(f"checkpoint {rel} not found locally or on S3")
        s3_download(_ckpt_key(rel), p)
    return p


def list_parts(rel_dir: str) -> list[str]:
    """Relative paths of part files under a checkpoint directory (local ∪ S3)."""
    names = set()
    d = ckpt_path(rel_dir)
    if d.exists():
        names |= {f.name for f in d.glob("*.parquet")}
    if C.s3_enabled():
        pref = _ckpt_key(rel_dir) + "/"
        for page in _s3().get_paginator("list_objects_v2").paginate(Bucket=C.S3_BUCKET, Prefix=pref):
            names |= {o["Key"].rsplit("/", 1)[-1] for o in page.get("Contents", []) if o["Key"].endswith(".parquet")}
    return sorted(f"{rel_dir}/{n}" for n in names)
