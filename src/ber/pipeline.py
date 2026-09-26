"""Stage functions shared by the notebooks — streaming / low-memory edition.

Designed to run the FULL data on a free-tier-class instance (2 vCPU, ~8 GB RAM):
  * cleaning streams the raw TSVs in chunks straight into Parquet;
  * candidate generation works one country block at a time with only that block's S1
    index in memory, streaming queries;
  * features are computed per country block and per chunk of pairs; training uses an
    entity-level subsample (train / valid roles), test pairs are scored in a stream and
    only (pair, probability) is kept.

Notebooks 00/02/03 call these for the *train* split and notebook 05 calls the very same
functions for the *test* split (identical cleaning, retrieval and features). Every stage
is checkpointed locally + on S3 and resumes from the last finished part.

Checkpoint layout (relative to CHECKPOINT_DIR, mirrored to s3://bucket/pipeline_artifacts*/):
    clean/{split}_{s1,s2,s3}.parquet        cleaned records (+ "row" = position in source)
    clean/train_gt.parquet                  ground truth
    candidates/{split}/{country}.parquet    (r_idx, s1_idx, ret, rk) per country block
    candidates/{split}_stats.npz            competition stats + split-level IDF vectors
    features/train/{country}-{k:04d}.parquet  labelled features of the train/valid entities
    scores/test/{country}-{k:04d}.parquet   (s1_idx, r_idx, prob) for every test pair
    model/lgbm.txt, model/meta.json

R ("records") = S2 rows followed by S3 rows; r_idx = global position in that order.
"""
from __future__ import annotations

import gc
import re
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import blocking, cleaning, features
from . import config as C
from . import io
from .metrics import explode_gt

SRC_COLS = ["entity_id", "business_name", "business_address", "country"]


def progress(stage: str, done: int, total: int, detail: str = "") -> None:
    """Write WORK_DIR/progress.json (read by scripts/dashboard.ps1). Never raises."""
    import json
    try:
        C.WORK_DIR.mkdir(parents=True, exist_ok=True)
        tmp = C.WORK_DIR / "progress.json.tmp"
        tmp.write_text(json.dumps({"stage": stage, "done": int(done), "total": int(max(total, 1)),
                                   "detail": detail, "ts": time.time()}))
        tmp.replace(C.WORK_DIR / "progress.json")
    except Exception:
        pass


def _safe(country: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", country.lower()) or "unknown"


# --------------------------------------------------------------------------------------
# Raw loading for the optional dev sample (full runs stream instead)
# --------------------------------------------------------------------------------------
def load_raw_split(split: str, sample_frac: float, seed: int = C.RANDOM_STATE) -> dict:
    """Self-consistent sub-world: a fraction of S1 entities, all their true matches, and the
    same fraction of the remaining S2/S3 records (distractor density per S1 preserved)."""
    raw = {k: io.read_raw(split, k, usecols=SRC_COLS) for k in ("s1", "s2", "s3")}
    if split == "train":
        raw["gt"] = io.read_raw(split, "gt")
    rng = np.random.default_rng(seed)
    s1 = raw["s1"]
    keep_s1 = s1[rng.random(len(s1)) < sample_frac]
    raw["s1"] = keep_s1.reset_index(drop=True)
    matched_keep, matched_all = set(), set()
    if "gt" in raw:
        pairs = explode_gt(raw["gt"])
        matched_all = set(pairs["candidate_entity_id"])
        ids = set(keep_s1["entity_id"])
        matched_keep = set(pairs.loc[pairs["source1_entity_id"].isin(ids), "candidate_entity_id"])
        raw["gt"] = raw["gt"][raw["gt"]["source1_entity_id"].isin(ids)].reset_index(drop=True)
    for k in ("s2", "s3"):
        d = raw[k]
        eid = d["entity_id"]
        keep = eid.isin(matched_keep).to_numpy() | (~eid.isin(matched_all).to_numpy() & (rng.random(len(d)) < sample_frac))
        raw[k] = d[keep].reset_index(drop=True)
    print("  sample:", {k: len(v) for k, v in raw.items()})
    return raw


# --------------------------------------------------------------------------------------
# Stage 1: cleaning (streamed)
# --------------------------------------------------------------------------------------
def _clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    return cleaning.clean_frame(df, n_jobs=1, chunk=len(df) + 1)


def _write_clean(frames, rel: str) -> int:
    """Write an iterable of cleaned frames into one Parquet file, adding the "row" column."""
    p = io.ckpt_path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    writer, n = None, 0
    for df in frames:
        df.insert(0, "row", np.arange(n, n + len(df), dtype=np.int32))
        n += len(df)
        progress(f"clean {rel}", n, n, f"{n:,} rows cleaned")
        t = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
        writer.write_table(t.cast(writer.schema))
        print(f"    {rel}: {n:,} rows cleaned", flush=True)
    writer.close()
    tmp.replace(p)
    io.save_file(p, rel)
    return n


def run_clean(split: str, force: bool = False) -> None:
    raw = None
    if split == "train":
        rel = "clean/train_gt.parquet"
        if force or not io.exists(rel):
            if C.SAMPLE_FRAC:
                raw = load_raw_split(split, C.SAMPLE_FRAC)
                gt = raw["gt"]
            else:
                gt = io.read_raw(split, "gt")
            io.save_parquet(gt, rel)
            del gt
    for k in ("s1", "s2", "s3"):
        rel = f"clean/{split}_{k}.parquet"
        if io.exists(rel) and not force:
            print(f"  [skip] {rel} exists")
            continue
        t = time.time()
        if C.SAMPLE_FRAC:
            if raw is None:
                raw = load_raw_split(split, C.SAMPLE_FRAC)
            frames = [cleaning.clean_frame(raw[k], n_jobs=C.N_JOBS, chunk=C.CLEAN_CHUNK_ROWS)]
            n = _write_clean(iter(frames), rel)
        else:
            reader = pd.read_csv(io.raw_path(split, k), usecols=SRC_COLS, chunksize=C.CLEAN_CHUNK_ROWS,
                                 **io.TSV_READ_KW)
            with Pool(C.N_JOBS) as pool:
                n = _write_clean(pool.imap(_clean_chunk, reader), rel)
        print(f"  cleaned {split}/{k}: {n:,} rows in {time.time() - t:.0f}s")
    del raw
    gc.collect()


# --------------------------------------------------------------------------------------
# Record access
# --------------------------------------------------------------------------------------
def _read(split: str, k: str, cols, country: str | None = None) -> pd.DataFrame:
    filt = [("country", "==", country)] if country is not None else None
    return pd.read_parquet(io.fetch(f"clean/{split}_{k}.parquet"), columns=cols, filters=filt)


def source_sizes(split: str) -> dict:
    return {k: pq.ParquetFile(io.fetch(f"clean/{split}_{k}.parquet")).metadata.num_rows for k in ("s1", "s2", "s3")}


def countries(split: str) -> list[str]:
    return sorted(pd.unique(_read(split, "s1", ["country"])["country"]).tolist())


def load_block(split: str, country: str, cols) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(S1, R) records of one country block with global indices in column "gidx"."""
    cols = list(dict.fromkeys(["row"] + list(cols)))
    n2 = source_sizes(split)["s2"]
    s1 = _read(split, "s1", cols, country).rename(columns={"row": "gidx"})
    r2 = _read(split, "s2", cols, country).rename(columns={"row": "gidx"})
    r3 = _read(split, "s3", cols, country).rename(columns={"row": "gidx"})
    r3["gidx"] = r3["gidx"] + n2
    r = pd.concat([r2, r3], ignore_index=True)
    return s1.reset_index(drop=True), r


def entity_ids(split: str) -> tuple[pd.Series, pd.Series]:
    """Global (S1 ids by s1_idx, R ids by r_idx)."""
    s1 = _read(split, "s1", ["entity_id"])["entity_id"]
    r = pd.concat([_read(split, "s2", ["entity_id"])["entity_id"],
                   _read(split, "s3", ["entity_id"])["entity_id"]], ignore_index=True)
    return s1, r


def load_true_pairs() -> pd.DataFrame:
    """Long-format labelled matches (source1_entity_id, candidate_entity_id) for train."""
    return explode_gt(io.load_parquet("clean/train_gt.parquet"))


def true_pair_keys(split: str = "train") -> tuple[np.ndarray, int]:
    """Sorted int64 keys r_idx * n_s1 + s1_idx of the labelled matches."""
    s1_ids, r_ids = entity_ids(split)
    tp = load_true_pairs()
    i1 = pd.Index(s1_ids).get_indexer(tp["source1_entity_id"])
    ir = pd.Index(r_ids).get_indexer(tp["candidate_entity_id"])
    ok = (i1 >= 0) & (ir >= 0)
    n1 = len(s1_ids)
    return np.sort(ir[ok].astype(np.int64) * n1 + i1[ok].astype(np.int64)), n1


def isin_sorted(keys: np.ndarray, sorted_ref: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(sorted_ref, keys)
    pos[pos >= len(sorted_ref)] = 0
    return sorted_ref[pos] == keys if len(sorted_ref) else np.zeros(len(keys), bool)


# --------------------------------------------------------------------------------------
# Stage 2: candidate generation
# --------------------------------------------------------------------------------------
def retrieval_max_df(n_s1: int) -> int | None:
    """Doc-frequency cap scaled with index size (keeps dev samples comparable)."""
    if C.RETRIEVAL_MAX_DF is None:
        return None
    return max(500, int(C.RETRIEVAL_MAX_DF * min(1.0, n_s1 / 2_200_000)))


def run_candidates(split: str, force: bool = False) -> dict:
    stats_rel = f"candidates/{split}_stats.npz"
    if io.exists(stats_rel) and not force:
        print(f"  [skip] {stats_rel} exists")
        return dict(np.load(io.fetch(stats_rel)))
    sizes = source_sizes(split)
    n1, nr = sizes["s1"], sizes["s2"] + sizes["s3"]
    st = {}
    for tag, n in (("r", nr), ("s1", n1)):
        for comp in ("", "_nm", "_ad"):          # combined score, name part, address part
            st[f"{tag}{comp}_top1"] = np.full(n, -1, np.float32)
            st[f"{tag}{comp}_top2"] = np.full(n, -1, np.float32)
        st[f"{tag}_n"] = np.zeros(n, np.int32)
        st[f"{tag}_nfreq"] = np.zeros(n, np.int32)   # how many S1 in the block share this exact name
        st[f"{tag}_afreq"] = np.zeros(n, np.int32)   # ... this exact (non-empty) address
    st["r_ntie"] = np.zeros(nr, np.int16)            # rivals within TIE_EPS of the record's best score
    df_n = np.zeros(blocking.N_FEATURES, np.int64)
    df_a = np.zeros(blocking.N_FEATURES, np.int64)
    max_df = retrieval_max_df(n1)
    schema = pa.schema([("r_idx", pa.int32()), ("s1_idx", pa.int32()), ("ret", pa.float32()),
                        ("ret_nm", pa.float32()), ("rk", pa.int8())])
    for country in countries(split):
        t0 = time.time()
        rel = f"candidates/{split}/{_safe(country)}.parquet"
        if io.exists(rel) and not force:
            # resume: block already retrieved -> rebuild its statistics from the saved pairs
            s1b, rb = load_block(split, country, ["name_core", "name_clean", "addr_clean"])
            _block_freqs(st, s1b, rb)
            df_n += blocking.doc_freq(blocking.hash_texts(
                "name", blocking.name_text(s1b["name_core"].tolist(), s1b["name_clean"].tolist()), C.N_JOBS))
            df_a += blocking.doc_freq(blocking.hash_texts("addr", s1b["addr_clean"].tolist(), C.N_JOBS))
            for batch in pq.ParquetFile(io.fetch(rel)).iter_batches(batch_size=2_000_000):
                b = batch.to_pandas()
                _accumulate(st, b["r_idx"].to_numpy().astype(np.int64), b["s1_idx"].to_numpy().astype(np.int64),
                            b["ret"].to_numpy(), b["ret_nm"].to_numpy(), ties=False)
            # second pass: near-ties need each record's final best score
            for batch in pq.ParquetFile(io.fetch(rel)).iter_batches(batch_size=2_000_000, columns=["r_idx", "ret"]):
                ri, v = batch.column("r_idx").to_numpy().astype(np.int64), batch.column("ret").to_numpy()
                np.add.at(st["r_ntie"], ri, (v >= st["r_top1"][ri] - C.TIE_EPS).astype(np.int16))
            print(f"  [{country}] resumed from {rel} ({time.time() - t0:.0f}s)", flush=True)
            del s1b, rb
            gc.collect()
            continue
        s1b, rb = load_block(split, country, TEXT_COLS)
        if len(rb) == 0:
            continue
        p = io.ckpt_path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        writer = pq.ParquetWriter(tmp, schema, compression="zstd")
        g1 = s1b["gidx"].to_numpy().astype(np.int64)
        gr = rb["gidx"].to_numpy().astype(np.int64)
        _block_freqs(st, s1b, rb)
        s1_texts = blocking.part_texts(*(s1b[c].tolist() for c in TEXT_COLS))
        del s1b
        gen = blocking.search_block(s1_texts, _QueryTexts(rb), C.RETRIEVAL_WEIGHTS, C.TOPK, C.MIN_SIM, max_df,
                                    C.N_JOBS, C.QUERY_CHUNK_ROWS, label=country)
        info = next(gen)
        del s1_texts
        df_n += info["df"]["name"]
        df_a += info["df"]["addr"]
        npairs = 0
        n_q = len(gr)
        for q, j, v, vn in gen:
            ri, si = gr[q], g1[j]
            _accumulate(st, ri, si, v, vn)
            progress(f"candidates {split}/{country}", int(q.max()) + 1 if len(q) else 0, n_q,
                     f"{npairs + len(q):,} pairs")
            rk = blocking.rank_within(q, v).astype(np.int8)
            writer.write_table(pa.table({"r_idx": ri.astype(np.int32), "s1_idx": si.astype(np.int32),
                                         "ret": v, "ret_nm": vn, "rk": rk}, schema=schema))
            npairs += len(q)
        writer.close()
        tmp.replace(p)
        io.save_file(p, rel)
        print(f"  [{country}] {len(g1):,} S1 x {len(gr):,} records -> {npairs:,} pairs "
              f"({npairs / max(1, len(gr)):.1f}/record) in {time.time() - t0:.0f}s", flush=True)
        del gen, rb
        gc.collect()
    st["idf_name"] = blocking.idf_from_df(df_n, n1)
    st["idf_addr"] = blocking.idf_from_df(df_a, n1)
    p = io.ckpt_path(stats_rel)
    np.savez(p, **st)
    io.save_file(p, stats_rel)
    return st


TEXT_COLS = ["name_core", "name_clean", "addr_clean", "name_phon", "addr_phon"]


def _block_freqs(st: dict, s1b: pd.DataFrame, rb: pd.DataFrame) -> None:
    """How many S1 records of the block share each exact name / (non-empty) address."""
    g1, gr = s1b["gidx"].to_numpy(), rb["gidx"].to_numpy()
    for col, key in (("name_core", "nfreq"), ("addr_clean", "afreq")):
        vc = s1b.loc[s1b[col] != "", col].value_counts()
        st[f"s1_{key}"][g1] = s1b[col].map(vc).fillna(0).to_numpy(np.int32)
        st[f"r_{key}"][gr] = rb[col].map(vc).fillna(0).to_numpy(np.int32)


def _accumulate(st: dict, ri: np.ndarray, si: np.ndarray, v: np.ndarray, vn: np.ndarray, ties: bool = True) -> None:
    """Fold one batch of retrieved pairs into the competition statistics. Every query's K
    results arrive in the same batch, so per-record values (best score, near-ties) are final."""
    va = (v - vn).astype(np.float32)
    for comp, val in (("", v), ("_nm", vn), ("_ad", va)):
        blocking.top2_update(st[f"r{comp}_top1"], st[f"r{comp}_top2"], ri, val)
        blocking.top2_update(st[f"s1{comp}_top1"], st[f"s1{comp}_top2"], si, val)
    np.add.at(st["r_n"], ri, 1)
    np.add.at(st["s1_n"], si, 1)
    if ties:
        best = st["r_top1"][ri]
        np.add.at(st["r_ntie"], ri, (v >= best - C.TIE_EPS).astype(np.int16))


class _QueryTexts:
    """Serve per-part query texts chunk by chunk (keeps RAM flat)."""

    def __init__(self, rb: pd.DataFrame):
        self.rb = rb
        self.n = len(rb)

    def __call__(self, a: int, b: int) -> dict:
        c = self.rb.iloc[a:b]
        return blocking.part_texts(*(c[col].tolist() for col in TEXT_COLS))


def candidate_files(split: str) -> list[str]:
    return [f"candidates/{split}/{_safe(c)}.parquet" for c in countries(split)
            if io.exists(f"candidates/{split}/{_safe(c)}.parquet")]


# --------------------------------------------------------------------------------------
# Entity roles for training (entity-disjoint subsample)
# --------------------------------------------------------------------------------------
def entity_roles(s1_ids: pd.Series) -> np.ndarray:
    """0 = unused, 1 = train, 2 = valid-A (early stopping + threshold), 3 = valid-B (report).
    A pure function of the entity id. The validation ranges come first, so changing the
    training fraction never moves the validation entities."""
    h = pd.util.hash_pandas_object(s1_ids.astype(str) + f"#{C.RANDOM_STATE}", index=False).to_numpy()
    u = (h % np.uint64(10_000)).astype(np.float64) / 100.0  # 0..100
    tr, va = C.train_valid_pct()
    role = np.zeros(len(s1_ids), np.int8)
    role[u < va / 2] = 2
    role[(u >= va / 2) & (u < va)] = 3
    role[(u >= va) & (u < va + tr)] = 1
    return role


def _pair_uniform(r_idx: np.ndarray, s1_idx: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-random number in [0, 1) per pair (reproducible subsampling)."""
    m = np.uint64(2**32)
    k = (r_idx.astype(np.uint64) * np.uint64(2654435761) + s1_idx.astype(np.uint64) * np.uint64(40503)) % m
    k = ((k ^ (k >> np.uint64(13))) * np.uint64(1274126177)) % m
    return k.astype(np.float64) / 2**32


# --------------------------------------------------------------------------------------
# Stage 3: features (per country block, per chunk)
# --------------------------------------------------------------------------------------
def iter_feature_chunks(split: str, stats: dict, keep_s1: np.ndarray | None = None, skip=None, filt=None):
    """Yield (part_name, meta, features) per chunk of candidate pairs.

    keep_s1: optional boolean mask over s1_idx; only those entities' pairs are featurised.
    skip   : optional callable(part_name) -> True to skip an already finished part.
    filt   : optional callable(batch_frame) -> boolean mask of pairs to featurise.
    """
    idf = {"name": stats["idf_name"], "addr": stats["idf_addr"]}
    sizes = source_sizes(split)
    for rel in candidate_files(split):
        country_tag = rel.rsplit("/", 1)[-1].removesuffix(".parquet")
        pf = pq.ParquetFile(io.fetch(rel))
        n_rows = pf.metadata.num_rows
        n_parts = -(-n_rows // C.FEATURE_CHUNK_ROWS)
        names = [f"{country_tag}-{k:04d}" for k in range(n_parts)]
        if skip is not None and all(skip(n) for n in names):
            print(f"  [skip] {country_tag}: all {n_parts} parts done")
            continue
        country = _read(split, "s1", ["country", "row"]).set_index("row")["country"]
        country = next(c for c in pd.unique(country) if _safe(c) == country_tag)
        t0 = time.time()
        s1b, rb = load_block(split, country, features.RECORD_COLS)
        pos1 = np.full(sizes["s1"], -1, np.int32)
        pos1[s1b["gidx"].to_numpy()] = np.arange(len(s1b), dtype=np.int32)
        posr = np.full(sizes["s2"] + sizes["s3"], -1, np.int32)
        posr[rb["gidx"].to_numpy()] = np.arange(len(rb), dtype=np.int32)
        print(f"  [{country}] records loaded ({time.time() - t0:.0f}s): {len(s1b):,} S1, {len(rb):,} R, "
              f"{n_rows:,} pairs in {n_parts} parts", flush=True)
        batches = pf.iter_batches(batch_size=C.FEATURE_CHUNK_ROWS)
        for k_part, (name, batch) in enumerate(zip(names, batches)):
            progress(f"{'scoring' if keep_s1 is None else 'features'} {split}/{country}", k_part, n_parts, name)
            if skip is not None and skip(name):
                continue
            t = time.time()
            b = batch.to_pandas()
            if keep_s1 is not None:
                b = b[keep_s1[b["s1_idx"].to_numpy()]]
            if filt is not None and len(b):
                b = b[filt(b)]
            if len(b) == 0:
                yield name, b[["s1_idx", "r_idx"]], pd.DataFrame()
                continue
            i1 = pos1[b["s1_idx"].to_numpy()]
            ir = posr[b["r_idx"].to_numpy()]
            f = features.pair_features(i1, ir, b["s1_idx"].to_numpy(), b["r_idx"].to_numpy(), b["ret"].to_numpy(),
                                       b["ret_nm"].to_numpy(), b["rk"].to_numpy(), s1b, rb, idf, stats,
                                       n_jobs=C.N_JOBS)
            print(f"    {name}: {len(b):,} pairs featurised in {time.time() - t:.0f}s", flush=True)
            yield name, b[["s1_idx", "r_idx"]].reset_index(drop=True), f
        del s1b, rb, pos1, posr
        gc.collect()


def run_features(split: str = "train", force: bool = False) -> list[str]:
    """Labelled features for the train / valid entities (role > 0) -> part files.

    Validation entities keep every candidate pair. Train entities keep all positives and
    hard negatives plus a weighted EASY_NEG_KEEP sample of easy negatives (column "weight")."""
    stats = dict(np.load(io.fetch(f"candidates/{split}_stats.npz")))
    s1_ids, _ = entity_ids(split)
    role = entity_roles(s1_ids)
    keys, n1 = true_pair_keys(split)
    print(f"  entities: train={int((role == 1).sum()):,} validA={int((role == 2).sum()):,} "
          f"validB={int((role == 3).sum()):,} of {len(role):,}")

    def easy_negative(ri: np.ndarray, si: np.ndarray, ret: np.ndarray) -> np.ndarray:
        lab = isin_sorted(ri.astype(np.int64) * n1 + si, keys)
        rel = ret / np.maximum(stats["r_top1"][ri], 1e-6)
        return (role[si] == 1) & ~lab & (rel < C.EASY_REL)

    def filt(b: pd.DataFrame) -> np.ndarray:
        ri, si = b["r_idx"].to_numpy(), b["s1_idx"].to_numpy()
        easy = easy_negative(ri, si, b["ret"].to_numpy())
        return ~easy | (_pair_uniform(ri, si) < C.EASY_NEG_KEEP)

    done = lambda name: (not force) and io.exists(f"features/{split}/{name}.parquet")
    for name, meta, f in iter_feature_chunks(split, stats, keep_s1=role > 0, skip=done, filt=filt):
        if len(meta) == 0:
            f = pd.DataFrame()
        else:
            ri, si = meta["r_idx"].to_numpy(), meta["s1_idx"].to_numpy()
            easy = easy_negative(ri, si, f["ret"].to_numpy())
            f.insert(0, "weight", np.where(easy, 1.0 / C.EASY_NEG_KEEP, 1.0).astype(np.float32))
            f.insert(0, "label", isin_sorted(ri.astype(np.int64) * n1 + si, keys).astype(np.int8))
            f.insert(0, "role", role[si])
            f.insert(0, "r_idx", ri)
            f.insert(0, "s1_idx", si)
        io.save_parquet(f, f"features/{split}/{name}.parquet")
    return io.list_parts(f"features/{split}")


def load_features(split: str = "train", columns=None) -> pd.DataFrame:
    parts = [p for p in io.list_parts(f"features/{split}")]
    dfs = [pd.read_parquet(io.fetch(p), columns=columns) for p in parts]
    return pd.concat([d for d in dfs if len(d)], ignore_index=True)


# --------------------------------------------------------------------------------------
# Stage 4/5: streamed scoring of every pair + submission files
# --------------------------------------------------------------------------------------
def run_scoring(split: str, model, force: bool = False) -> list[str]:
    stats = dict(np.load(io.fetch(f"candidates/{split}_stats.npz")))
    feats = model.feature_name()
    done = lambda name: (not force) and io.exists(f"scores/{split}/{name}.parquet")
    for name, meta, f in iter_feature_chunks(split, stats, skip=done):
        prob = model.predict(f[feats].to_numpy(dtype=np.float32), num_threads=C.N_JOBS)
        out = meta.assign(prob=prob.astype(np.float32))
        io.save_parquet(out, f"scores/{split}/{name}.parquet", sync=True)
    return io.list_parts(f"scores/{split}")


def load_scores(split: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(io.fetch(p)) for p in io.list_parts(f"scores/{split}")], ignore_index=True)


def decide_idx(scores: pd.DataFrame, threshold: float, one_to_one: bool) -> pd.DataFrame:
    """Final matches on integer ids: optional one-to-one per record, then prob >= threshold."""
    s = scores[scores["prob"] >= threshold]
    if one_to_one:
        s = s.sort_values(["r_idx", "prob"], ascending=[True, False]).drop_duplicates("r_idx")
    return s


def write_submission(split: str, threshold: float, one_to_one: bool, out_dir) -> dict:
    """Write matching_results.tsv + candidate_pairs.tsv, one country block at a time.

    Every S1 entity gets exactly one row in each file (empty list allowed); lists hold unique
    comma-separated S2/S3 ids; the candidate list is exactly the set of pairs the model scored,
    so every matched id is also a candidate.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fm = open(out_dir / "matching_results.tsv", "w", encoding="utf-8", newline="\n")
    fc = open(out_dir / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n")
    fm.write("source1_entity_id\tmatched_entity_ids\n")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    parts = io.list_parts(f"scores/{split}")
    n = {"s1_rows": 0, "matched_s1": 0, "matches": 0, "candidates": 0}
    for country in countries(split):
        tag = _safe(country)
        s1b, rb = load_block(split, country, ["entity_id"])
        mine = [p for p in parts if p.rsplit("/", 1)[-1].startswith(tag + "-")]
        empty = pd.DataFrame({"s1_idx": np.array([], np.int32), "r_idx": np.array([], np.int32),
                              "prob": np.array([], np.float32)})
        sc = pd.concat([pd.read_parquet(io.fetch(p)) for p in mine], ignore_index=True) if mine else empty
        dec = decide_idx(sc, threshold, one_to_one)
        rid = pd.Series(rb["entity_id"].to_numpy(), index=rb["gidx"].to_numpy())
        s1_gidx = s1b["gidx"].to_numpy()
        s1_eid = s1b["entity_id"].tolist()
        for fh, pairs in ((fc, sc), (fm, dec)):
            order = np.lexsort((pairs["r_idx"].to_numpy(), pairs["s1_idx"].to_numpy()))
            si = pairs["s1_idx"].to_numpy()[order]
            ids = rid.reindex(pairs["r_idx"].to_numpy()[order]).to_numpy()
            lo = np.searchsorted(si, s1_gidx, "left")
            hi = np.searchsorted(si, s1_gidx, "right")
            for eid, a, b in zip(s1_eid, lo, hi):
                fh.write(f"{eid}\t{','.join(ids[a:b])}\n" if b > a else f"{eid}\t\n")
        n["s1_rows"] += len(s1b)
        n["matched_s1"] += int(dec["s1_idx"].nunique())
        n["matches"] += len(dec)
        n["candidates"] += len(sc)
        print(f"  [{country}] {len(s1b):,} S1 | {len(sc):,} candidates | {len(dec):,} matches", flush=True)
        del sc, dec, rid, s1b, rb
        gc.collect()
    fm.close()
    fc.close()
    print(f"  wrote {out_dir}: {n}")
    return n


def load_training_matrices(split: str = "train", roles=(1, 2, 3)) -> dict:
    """Features of the train / valid entities as per-role float32 matrices, built part by part
    without ever materialising one big DataFrame (peak RAM ~ size of the train matrix).

    Returns {"features": [...], role: {"X", "y", "s1_idx", "r_idx", "w"}} for the requested roles
    (load train + valid-A first, valid-B later, to keep peak RAM low on 8 GB machines).
    """
    parts = io.list_parts(f"features/{split}")
    meta_cols = ["s1_idx", "r_idx", "role", "label"]
    counts, feats = {r_: 0 for r_ in roles}, None
    for p in parts:  # pass 1: sizes + feature list (reads only two small columns)
        pf = pq.ParquetFile(io.fetch(p))
        if pf.metadata.num_rows == 0:
            continue
        if feats is None:
            feats = [c for c in pf.schema_arrow.names if c not in features.NON_FEATURES]
        role = pf.read(columns=["role"]).column("role").to_numpy()
        for r_ in roles:
            counts[r_] += int((role == r_).sum())
    out = {"features": feats}
    for r_ in roles:
        out[r_] = {"X": np.empty((counts[r_], len(feats)), np.float32), "y": np.empty(counts[r_], np.int8),
                   "s1_idx": np.empty(counts[r_], np.int32), "r_idx": np.empty(counts[r_], np.int32),
                   "w": np.ones(counts[r_], np.float32)}
    fill = {r_: 0 for r_ in roles}
    for p in parts:  # pass 2: route rows into the per-role matrices
        pf = pq.ParquetFile(io.fetch(p))
        if pf.metadata.num_rows == 0:
            continue
        has_w = "weight" in pf.schema_arrow.names
        t = pf.read(columns=meta_cols + feats + (["weight"] if has_w else []))
        role = t.column("role").to_numpy()
        for r_ in roles:
            m = role == r_
            n = int(m.sum())
            if n == 0:
                continue
            a, b = fill[r_], fill[r_] + n
            d = out[r_]
            for k, c in enumerate(feats):
                d["X"][a:b, k] = t.column(c).to_numpy(zero_copy_only=False)[m]
            d["y"][a:b] = t.column("label").to_numpy()[m]
            d["s1_idx"][a:b] = t.column("s1_idx").to_numpy()[m]
            d["r_idx"][a:b] = t.column("r_idx").to_numpy()[m]
            if has_w:
                d["w"][a:b] = t.column("weight").to_numpy()[m]
            fill[r_] = b
        del t
        gc.collect()
    names = {1: "train", 2: "validA", 3: "validB"}
    print("  matrices: " + " | ".join(f"{names[r_]} {counts[r_]:,}" for r_ in roles)
          + f" rows x {len(feats)} features ({sum(out[r_]['X'].nbytes for r_ in roles) / 1e9:.2f} GB)")
    return out


def sample_features(split: str = "train", frac: float = 0.05, columns=None, seed: int = 0) -> pd.DataFrame:
    """A random row sample of the feature parts (for summaries / plots without loading everything)."""
    rng = np.random.default_rng(seed)
    out = []
    for p in io.list_parts(f"features/{split}"):
        t = pd.read_parquet(io.fetch(p), columns=columns)
        if len(t):
            out.append(t.sample(frac=frac, random_state=int(rng.integers(1 << 31))))
    return pd.concat(out, ignore_index=True)
