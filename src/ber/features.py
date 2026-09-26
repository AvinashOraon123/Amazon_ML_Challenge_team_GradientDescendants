"""Pairwise similarity features for (S1 record, S2/S3 record) candidate pairs.

All features are country-agnostic (no country one-hot): the test set contains a country
(France) never seen in training, so the model must judge matches purely from how similar
two records are, not from where they are.

Feature families
----------------
name_*   : string similarity on cleaned names (rapidfuzz C++ scorers, multithreaded)
legal_*  : agreement of canonical legal forms (llc / pvt ltd / sarl ...)
addr_*   : string similarity on cleaned addresses
num_*    : agreement of numeric address tokens (house / plot / flat numbers), postcode, state
ret / rk : retrieval cosine and rank of this S1 among the record's neighbours
grp_*    : "competition" features — how this pair's retrieval score compares with the best /
           second-best rival for the same S2/S3 record and for the same S1 entity. The
           single most important signal for precision: a pair that is similar in absolute
           terms but clearly beaten by another S1 is almost never a match.

Implementation notes (built for 2 vCPU / 8 GB): everything is computed per chunk of pairs;
token-set statistics (Jaccard, overlap, containment) are sparse binary-hash dot products
instead of Python set loops; TF-IDF cosines re-hash only the chunk's strings using the
split-level IDF vectors saved by the candidate stage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from .blocking import VEC

FLOAT = np.float32
_TOK = HashingVectorizer(analyzer="word", token_pattern=r"\S+", n_features=2**20, binary=True,
                         norm=None, alternate_sign=False, lowercase=False, dtype=np.float32)

# Columns of the cleaned record tables needed to compute features.
RECORD_COLS = ["entity_id", "name_core", "name_clean", "name_alt", "legal", "addr_clean",
               "addr_nums", "state", "postcode", "name_len", "name_ntok", "addr_ntok",
               "name_nonlatin", "name_is_domain", "name_phon", "addr_phon"]


def _cp(a, b, scorer, n_jobs) -> np.ndarray:
    return cpdist(a, b, scorer=scorer, workers=n_jobs, dtype=np.float32).astype(FLOAT)


def _rowdot(A: sp.csr_matrix, B: sp.csr_matrix) -> np.ndarray:
    return np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(FLOAT)


def _tfidf(kind: str, texts: list[str], idf: np.ndarray) -> sp.csr_matrix:
    X = VEC[kind].transform(texts).tocsr()
    X.data = np.log1p(X.data) * idf[X.indices]
    return normalize(X, copy=False)


def _set_stats(a: list[str], b: list[str]):
    """|A∩B|, |A|, |B| of whitespace-token sets for aligned lists (sparse, vectorised)."""
    A, B = _TOK.transform(a), _TOK.transform(b)
    inter = _rowdot(A, B)
    na = np.diff(A.indptr).astype(FLOAT)
    nb = np.diff(B.indptr).astype(FLOAT)
    return inter, na, nb


def _jacc(inter, na, nb):
    union = na + nb - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, np.nan).astype(FLOAT)


def _take(df: pd.DataFrame, col: str, idx: np.ndarray) -> list:
    return df[col].iloc[idx].tolist()


def retrieval_features(g1: np.ndarray, gr: np.ndarray, ret: np.ndarray, ret_nm: np.ndarray,
                       rk: np.ndarray, stats: dict) -> dict:
    """Cheap features available right after retrieval (no string comparisons): retrieval score and
    its name / address parts, rank, competition per record and per S1, near-ties, ambiguity.
    Used by the candidate pruner and as the first block of the full matcher's features."""
    f = {}
    ret_ad = (ret - ret_nm).astype(FLOAT)
    f["ret"], f["ret_nm"], f["ret_ad"] = ret.astype(FLOAT), ret_nm.astype(FLOAT), ret_ad
    f["rk"] = rk.astype(np.int16)
    for tag, idx in (("r", gr), ("s1", g1)):
        f[f"grp_{tag}_n"] = stats[f"{tag}_n"][idx].astype(np.int32)
        for comp, val in (("", ret), ("_nm", ret_nm), ("_ad", ret_ad)):
            t1, t2 = stats[f"{tag}{comp}_top1"][idx], stats[f"{tag}{comp}_top2"][idx]
            f[f"grp_{tag}{comp}_rel"] = (val / np.maximum(t1, 1e-6)).astype(FLOAT)
            # margin to the best OTHER candidate (positive => this pair is the winner)
            f[f"grp_{tag}{comp}_margin"] = np.where(val >= t1, val - np.maximum(t2, 0), val - t1).astype(FLOAT)
        f[f"grp_{tag}_gap12"] = (stats[f"{tag}_top1"][idx] - np.maximum(stats[f"{tag}_top2"][idx], 0)).astype(FLOAT)
    f["grp_r_ntie"] = stats["r_ntie"][gr].astype(np.int16)
    # ambiguity: how many S1 businesses in the block carry exactly this name / address
    f["freq_s1_name"] = stats["s1_nfreq"][g1]
    f["freq_r_name"] = stats["r_nfreq"][gr]
    f["freq_s1_addr"] = stats["s1_afreq"][g1]
    f["freq_r_addr"] = stats["r_afreq"][gr]
    return f


def pair_features(i1: np.ndarray, ir: np.ndarray, g1: np.ndarray, gr: np.ndarray, ret: np.ndarray,
                  ret_nm: np.ndarray, rk: np.ndarray, s1: pd.DataFrame, r: pd.DataFrame, idf: dict,
                  stats: dict, n_jobs: int = 2) -> pd.DataFrame:
    """Feature frame for aligned arrays of pairs.

    i1 / ir : positions of the pair's records in the block tables `s1` / `r` (RECORD_COLS)
    g1 / gr : global s1_idx / r_idx of the pair (index the split-level `stats` arrays)
    ret, ret_nm : combined retrieval cosine and its name contribution (address = ret - ret_nm)
    idf     : {"name": idf vector, "addr": idf vector} (split-level, from the candidate stage)
    stats   : competition / ambiguity statistics from the candidate stage
    """
    f = retrieval_features(g1, gr, ret, ret_nm, rk, stats)

    # ---------------- names ----------------
    na, nb = _take(s1, "name_core", i1), _take(r, "name_core", ir)
    f["name_ratio"] = _cp(na, nb, fuzz.ratio, n_jobs)
    f["name_partial"] = _cp(na, nb, fuzz.partial_ratio, n_jobs)
    f["name_tsort"] = _cp(na, nb, fuzz.token_sort_ratio, n_jobs)
    f["name_tset"] = _cp(na, nb, fuzz.token_set_ratio, n_jobs)
    f["name_jw"] = _cp(na, nb, JaroWinkler.normalized_similarity, n_jobs)
    fa, fb = _take(s1, "name_clean", i1), _take(r, "name_clean", ir)
    f["name_full_tset"] = _cp(fa, fb, fuzz.token_set_ratio, n_jobs)
    # concatenated forms: "victorylaboratories" (domain) vs "victory laboratories"
    f["name_nospace"] = _cp([x.replace(" ", "") for x in na], [x.replace(" ", "") for x in nb], fuzz.ratio, n_jobs)
    # alias side of "X fka Y" / "X dba Y" on either record (computed only where present)
    alt_a, alt_b = _take(s1, "name_alt", i1), _take(r, "name_alt", ir)
    has = np.fromiter((bool(x) or bool(y) for x, y in zip(alt_a, alt_b)), dtype=bool, count=len(na))
    best = np.full(len(na), np.nan, dtype=FLOAT)
    if has.any():
        h = np.flatnonzero(has)
        a_, b_ = [na[i] for i in h], [nb[i] for i in h]
        aa_ = [alt_a[i] or na[i] for i in h]
        bb_ = [alt_b[i] or nb[i] for i in h]
        best[h] = np.maximum(_cp(a_, bb_, fuzz.token_set_ratio, n_jobs), _cp(aa_, b_, fuzz.token_set_ratio, n_jobs))
    f["name_alt_best"] = best
    nm_a = [c or x for c, x in zip(na, fa)]
    nm_b = [c or x for c, x in zip(nb, fb)]
    f["name_cos"] = _rowdot(_tfidf("name", nm_a, idf["name"]), _tfidf("name", nm_b, idf["name"]))
    inter, ca, cb = _set_stats(na, nb)
    f["name_jacc"], f["name_tok_overlap"] = _jacc(inter, ca, cb), inter
    f["name_first_eq"] = np.fromiter(((x.split(" ", 1)[0] == y.split(" ", 1)[0]) and bool(x) for x, y in zip(na, nb)),
                                     dtype=np.int8, count=len(na))
    f["name_exact"] = np.fromiter((x == y and bool(x) for x, y in zip(na, nb)), dtype=np.int8, count=len(na))
    la = s1["name_len"].to_numpy()[i1].astype(FLOAT)
    lb = r["name_len"].to_numpy()[ir].astype(FLOAT)
    f["name_len_ratio"] = (np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)).astype(FLOAT)
    f["name_ntok_a"] = s1["name_ntok"].to_numpy()[i1]
    f["name_ntok_b"] = r["name_ntok"].to_numpy()[ir]
    li, lca, lcb = _set_stats(_take(s1, "legal", i1), _take(r, "legal", ir))
    f["legal_jacc"] = np.where((lca > 0) & (lcb > 0), _jacc(li, lca, lcb), np.nan).astype(FLOAT)
    # phonetic skeletons: robust to phonetic transliteration ("brait prodyusr" ~ "bright producer")
    pa_, pb_ = _take(s1, "name_phon", i1), _take(r, "name_phon", ir)
    f["name_phon_ratio"] = _cp(pa_, pb_, fuzz.ratio, n_jobs)
    f["name_phon_tset"] = _cp(pa_, pb_, fuzz.token_set_ratio, n_jobs)
    f["name_phon_nospace"] = _cp([x.replace(" ", "") for x in pa_], [x.replace(" ", "") for x in pb_], fuzz.ratio, n_jobs)
    f["b_nonlatin"] = r["name_nonlatin"].to_numpy()[ir]
    f["b_is_domain"] = r["name_is_domain"].to_numpy()[ir]

    # ---------------- addresses ----------------
    aa, ab = _take(s1, "addr_clean", i1), _take(r, "addr_clean", ir)
    a_empty = np.fromiter((not x for x in aa), dtype=bool, count=len(aa))
    b_empty = np.fromiter((not x for x in ab), dtype=bool, count=len(ab))
    any_empty = a_empty | b_empty
    for nm, sc in (("addr_ratio", fuzz.ratio), ("addr_tset", fuzz.token_set_ratio),
                   ("addr_tsort", fuzz.token_sort_ratio), ("addr_partial", fuzz.partial_ratio)):
        f[nm] = np.where(any_empty, np.nan, _cp(aa, ab, sc, n_jobs)).astype(FLOAT)
    f["addr_cos"] = np.where(any_empty, np.nan,
                             _rowdot(_tfidf("addr", aa, idf["addr"]), _tfidf("addr", ab, idf["addr"]))).astype(FLOAT)
    qa_, qb_ = _take(s1, "addr_phon", i1), _take(r, "addr_phon", ir)
    f["addr_phon_tset"] = np.where(any_empty, np.nan, _cp(qa_, qb_, fuzz.token_set_ratio, n_jobs)).astype(FLOAT)
    inter, ca, cb = _set_stats(aa, ab)
    f["addr_jacc"] = np.where(any_empty, np.nan, _jacc(inter, ca, cb)).astype(FLOAT)
    f["addr_tok_overlap"] = inter
    mn = np.minimum(ca, cb)
    # containment: share of the shorter address's tokens found in the longer one
    f["addr_contain"] = np.where(mn > 0, inter / np.maximum(mn, 1), np.nan).astype(FLOAT)
    f["addr_empty_a"] = a_empty.astype(np.int8)
    f["addr_empty_b"] = b_empty.astype(np.int8)
    f["addr_ntok_a"] = s1["addr_ntok"].to_numpy()[i1]
    f["addr_ntok_b"] = r["addr_ntok"].to_numpy()[ir]

    # ---------------- numbers / postcode / state ----------------
    xa, xb = _take(s1, "addr_nums", i1), _take(r, "addr_nums", ir)
    inter, ca, cb = _set_stats(xa, xb)
    both = (ca > 0) & (cb > 0)
    f["num_jacc"] = np.where(both, _jacc(inter, ca, cb), np.nan).astype(FLOAT)
    f["num_shared"] = inter
    f["num_conflict"] = (both & (inter == 0)).astype(np.int8)
    f["num_first_eq"] = np.where(both, np.fromiter((x.split(" ", 1)[0] == y.split(" ", 1)[0] for x, y in zip(xa, xb)),
                                                   dtype=FLOAT, count=len(xa)), np.nan).astype(FLOAT)
    pa, pb = np.asarray(_take(s1, "postcode", i1), dtype=object), np.asarray(_take(r, "postcode", ir), dtype=object)
    both_pc = (pa != "") & (pb != "")
    f["postcode_eq"] = np.where(both_pc, (pa == pb).astype(FLOAT), np.nan).astype(FLOAT)
    sa, sb = np.asarray(_take(s1, "state", i1), dtype=object), np.asarray(_take(r, "state", ir), dtype=object)
    both_st = (sa != "") & (sb != "")
    f["state_eq"] = np.where(both_st, (sa == sb).astype(FLOAT), np.nan).astype(FLOAT)
    f["b_is_s3"] = np.fromiter((e.startswith("S3-") for e in _take(r, "entity_id", ir)), dtype=np.int8, count=len(ir))
    return pd.DataFrame(f)


# Everything except identifiers / labels / bookkeeping is a model feature.
NON_FEATURES = {"s1_idx", "r_idx", "label", "source1_entity_id", "candidate_entity_id", "fold", "role", "prob", "weight"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]
