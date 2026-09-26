"""Blocking + TF-IDF candidate generation — memory-lean version for small instances.

Strategy
--------
1. **Hard block on country.** 100% of labelled true pairs share a country, so this is
   lossless. The block key is the raw country string, so an unseen country (France, in
   test) simply forms its own block.
2. **Within a block, sparse TF-IDF nearest neighbours from each S2/S3 record ("query")
   into S1 ("index")** on one combined vector per record:
       name part : char 3-grams (word-boundary) of name_core -> typo/transliteration tolerant
       addr part : word 1-2 grams of addr_clean              -> catches renamed records
   The two parts are l2-normalised separately and concatenated with equal weight, so
   cos(combo) = ½·cos(name) + ½·cos(addr) when both fields exist and degrades gracefully
   when one is empty. On the dev sample this single channel with K=10 had higher recall
   than the union of separate name/address/combo channels, at the same cost.
   Query direction matters: each S2/S3 record matches at most ONE S1 entity (01_eda), so
   its true partner only needs to be in its own top-K.
3. N-grams present in more than `max_df` index rows are dropped from the retrieval
   vectors: they carry little identity signal and dominate sparse-matmul cost.

Memory profile: only the S1 index of the current block is materialised; queries are
hashed, weighted and searched chunk by chunk. IDF comes from the block's S1 records.
Competition statistics (best / second-best retrieval score per query and per S1 entity)
are accumulated on the fly into small arrays, so later stages can derive "how does this
pair compare with its rivals" features without ever grouping the full pair table.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

N_FEATURES = 2**21

VEC = {
    "name": HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=N_FEATURES,
                              alternate_sign=False, norm=None, lowercase=False, dtype=np.float32),
    "addr": HashingVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+",
                              n_features=N_FEATURES, alternate_sign=False, norm=None,
                              lowercase=False, dtype=np.float32),
}
VEC["phon"] = VEC["name"]    # char 3-grams of the phonetic skeleton of the name
VEC["addrp"] = VEC["addr"]   # word 1-2 grams of the phonetic skeleton of the address
NAME_PARTS = ("name", "phon")  # parts whose contribution is reported as the "name" score
PART_ORDER = ("name", "phon", "addr", "addrp")
PART_DF_FACTOR = {"name": 1.0, "phon": 1.0, "addr": 1.0, "addrp": 1.0}  # per-part multiplier of max_df


def part_texts(name_core, name_clean, addr_clean, name_phon, addr_phon) -> dict:
    """Retrieval texts per part for aligned lists of records."""
    return {"name": name_text(name_core, name_clean), "phon": list(name_phon),
            "addr": list(addr_clean), "addrp": list(addr_phon)}


def name_text(name_core, name_clean) -> list[str]:
    """Text for the name channel: core name, falling back to the full cleaned name."""
    return [c if c else f for c, f in zip(name_core, name_clean)]


def _hash(kind: str, texts: list[str]) -> sp.csr_matrix:
    X = VEC[kind].transform(texts).tocsr()
    X.data = np.log1p(X.data)  # sublinear TF
    return X


def hash_texts(kind: str, texts: list[str], n_jobs: int, chunk: int = 200_000) -> sp.csr_matrix:
    if n_jobs <= 1 or len(texts) <= chunk:
        return _hash(kind, texts)
    parts = Parallel(n_jobs=n_jobs)(  # processes: the hashing is GIL-bound Python
        delayed(_hash)(kind, texts[i : i + chunk]) for i in range(0, len(texts), chunk))
    return sp.vstack(parts, format="csr")


def doc_freq(X: sp.csr_matrix) -> np.ndarray:
    return np.bincount(X.indices, minlength=N_FEATURES).astype(np.int64)


def idf_from_df(df: np.ndarray, n_docs: int) -> np.ndarray:
    return (np.log((1 + n_docs) / (1 + df)) + 1).astype(np.float32)


def weight(X: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    """TF-IDF + l2 normalisation (in place on a fresh hashed matrix)."""
    X.data *= idf[X.indices]
    return normalize(X, norm="l2", copy=False)


def combo(parts: list[sp.csr_matrix], weights: list[float]) -> sp.csr_matrix:
    """Concatenate l2-normalised parts with weights w_k (sum 1):
    cos(combo) = sum_k w_k * cos_k when every part is present, degrading gracefully when a
    record has an empty field."""
    return normalize(sp.hstack([X * np.float32(np.sqrt(w)) for X, w in zip(parts, weights)], format="csr"), copy=False)


def _prune(X: sp.csr_matrix, keep: np.ndarray | None) -> sp.csr_matrix:
    if keep is None:
        return X
    X = X.copy()
    X.data *= keep[X.indices]
    X.eliminate_zeros()
    return normalize(X, copy=False)


def rank_within(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """1-based rank of each (q, v) among rows with the same q, by descending v."""
    order = np.lexsort((-v, q))
    ranks = np.empty(len(q), dtype=np.int16)
    qs = q[order]
    start = np.r_[0, np.flatnonzero(np.diff(qs)) + 1]
    run = np.diff(np.r_[start, len(qs)])
    ranks[order] = (np.arange(len(qs)) - np.repeat(start, run) + 1).astype(np.int16)
    return ranks


def top2_update(top1: np.ndarray, top2: np.ndarray, keys: np.ndarray, vals: np.ndarray) -> None:
    """Fold a batch of (key, value) into running per-key best / second-best (in place)."""
    if len(keys) == 0:
        return
    order = np.lexsort((-vals, keys))
    k, v = keys[order], vals[order]
    start = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
    size = np.diff(np.r_[start, len(k)])
    uk = k[start]
    b1 = v[start]
    b2 = np.where(size > 1, v[np.minimum(start + 1, len(v) - 1)], -1.0).astype(np.float32)
    o1, o2 = top1[uk], top2[uk]
    top1[uk] = np.maximum(o1, b1)
    top2[uk] = np.maximum(np.minimum(o1, b1), np.maximum(o2, b2))


def _name_part(Q: sp.csr_matrix, index: sp.csr_matrix, q: np.ndarray, j: np.ndarray, n_name_cols: int) -> np.ndarray:
    """Contribution of the NAME parts of the vectors to cos(Q[q], index[j]) for each pair
    (the address contribution is ret - name part)."""
    P = Q[q].multiply(index[j]).tocsr()
    w = np.where(P.indices < n_name_cols, P.data, 0.0)
    out = np.zeros(P.shape[0], dtype=np.float32)
    nz = np.diff(P.indptr) > 0
    if nz.any():
        out[nz] = np.add.reduceat(w, P.indptr[:-1][nz])
    return out


def _vectorise(texts: dict, parts: list[str], weights: list[float], idfs: dict, n_jobs: int) -> sp.csr_matrix:
    return combo([weight(hash_texts(VEC_KIND[p], texts[p], n_jobs), idfs[p]) for p in parts], weights)


VEC_KIND = {"name": "name", "phon": "name", "addr": "addr", "addrp": "addr"}


def search_block(s1_texts: dict, q_texts, weights: dict, k: int, min_sim: float, max_df: int | None,
                 n_jobs: int, chunk: int, label: str = ""):
    """Yield (q_local, s1_local, cos, name_part) arrays per chunk of queries against one
    block's S1 index. s1_texts: {part: list}; q_texts: callable(start, stop) -> {part: list}.
    The first yielded item is {"idf": {part: idf}, "df": {part: df}} of the block.
    """
    from sparse_dot_topn import sp_matmul_topn

    t = time.time()
    parts = [p for p in PART_ORDER if weights.get(p, 0) > 0]
    w = [weights[p] for p in parts]
    n_s1 = len(s1_texts[parts[0]])
    mats, dfs, idfs = [], {}, {}
    for p in parts:
        X = hash_texts(VEC_KIND[p], s1_texts[p], n_jobs)
        dfs[p] = doc_freq(X)
        idfs[p] = idf_from_df(dfs[p], n_s1)
        mats.append(weight(X, idfs[p]))
    # per-part cap: the phonetic skeletons are short, so their n-grams are far more common
    caps = {p: (None if max_df is None else int(max_df * PART_DF_FACTOR.get(p, 1.0))) for p in parts}
    keep = None if max_df is None else np.concatenate([dfs[p] <= caps[p] for p in parts]).astype(np.float32)
    index = _prune(combo(mats, w), keep)
    del mats
    indexT = index.T.tocsr()
    n_name_cols = N_FEATURES * sum(p in NAME_PARTS for p in parts)
    print(f"  [{label}] index: {n_s1:,} S1, parts={parts}, nnz/row={indexT.nnz / max(1, n_s1):.1f} "
          f"({time.time() - t:.0f}s)", flush=True)
    yield {"idf": idfs, "df": dfs}

    n_q = q_texts.n
    for s in range(0, n_q, chunk):
        t = time.time()
        Q = _prune(_vectorise(q_texts(s, s + chunk), parts, w, idfs, n_jobs), keep)
        M = sp_matmul_topn(Q, indexT, top_n=k, threshold=min_sim, sort=True, n_threads=n_jobs).tocoo()
        q, j = M.row.astype(np.int64), M.col.astype(np.int64)
        yield (q + s, j, M.data.astype(np.float32), _name_part(Q, index, q, j, n_name_cols))
        print(f"  [{label}] queries {min(s + chunk, n_q):,}/{n_q:,} ({time.time() - t:.1f}s/chunk)", flush=True)
