"""Stage 4: pair features for every candidate (Source-1 entity, Source-2/3 record).

Feature groups (country is deliberately NOT a feature, so the matcher transfers to unseen countries):
  * embedding: cosine in the joint / address / name views, the pair's rank in the record's neighbour
    list per view, the record's best and 2nd-best score per view and the gap to them
  * string (rapidfuzz, on normalised text): several fuzzy ratios for name, core name, skeleton, address
  * numbers: house/unit/postcode agreement, incl. dropped-leading-digit and zero-padding variants
  * acronym / concatenation checks for names such as 'SHLA' or 'energyprivate'
  * context: candidate-set sizes, record's rank among the entity's candidates
"""
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .normalize import MARKERS

VIEWS = ("z", "za", "zn")


def load_embeddings(work_dir):
    return {v: np.load(Path(work_dir) / f"emb_{v}.npy", mmap_mode="r") for v in VIEWS}


def _rowdot(a, b, i, j, chunk=2_000_000):
    out = np.empty(len(i), np.float32)
    for s in range(0, len(i), chunk):
        out[s:s + chunk] = (a[i[s:s + chunk]].astype(np.float32) * b[j[s:s + chunk]].astype(np.float32)).sum(1)
    return out


def embedding_features(cand, nb, embs, n_records):
    """cand: DataFrame(s1_rid, rid). nb: neighbours dict. Returns dict of numpy feature columns."""
    s1 = cand["s1_rid"].to_numpy().astype(np.int64)
    r = cand["rid"].to_numpy().astype(np.int64)
    qpos = np.full(n_records, -1, np.int64)
    qpos[nb["q"]] = np.arange(len(nb["q"]))
    qi = qpos[r]
    f = {}
    for v in VIEWS:
        cos = _rowdot(embs[v], embs[v], s1, r)
        nbr = nb[f"nbr_{v}"]
        sc = nb[f"sc_{v}"].astype(np.float32)
        K = nbr.shape[1]
        rank = np.full(len(r), K + 1, np.int16)
        for c in range(0, len(r), 2_000_000):
            sl = slice(c, c + 2_000_000)
            hit = nbr[qi[sl]] == s1[sl, None]
            rank[sl] = np.where(hit.any(1), hit.argmax(1) + 1, K + 1)
        top1, top2 = sc[qi, 0], sc[qi, 1]
        # views searched only for some records (name view: address-less records) have no scores -> -1
        top1 = np.where(np.isfinite(top1), top1, -1.0).astype(np.float32)
        top2 = np.where(np.isfinite(top2), top2, -1.0).astype(np.float32)
        f[f"{v}_cos"] = cos
        f[f"{v}_rank"] = rank.astype(np.float32)
        f[f"{v}_top1"] = top1
        f[f"{v}_gap1"] = top1 - cos                      # 0 when this S1 is the record's best
        f[f"{v}_margin12"] = top1 - top2                  # how decisive the record's best neighbour is
        f[f"{v}_vs2"] = cos - np.where(rank == 1, top2, top1)  # lead over the best OTHER entity
    return f


def _cp(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / (1.0 if scorer is JaroWinkler.normalized_similarity else 100.0)


def string_features(p):
    """p: DataFrame with columns <col>_1 (Source-1 side) and <col>_2 (record side)."""
    f = {}
    n1, n2 = p["name_n_1"].to_list(), p["name_n_2"].to_list()
    c1, c2 = p["core_n_1"].to_list(), p["core_n_2"].to_list()
    k1, k2 = p["skel_1"].to_list(), p["skel_2"].to_list()
    a1, a2 = p["addr_n_1"].to_list(), p["addr_n_2"].to_list()
    f["name_ratio"] = _cp(fuzz.ratio, n1, n2)
    f["name_tsort"] = _cp(fuzz.token_sort_ratio, n1, n2)
    f["name_tset"] = _cp(fuzz.token_set_ratio, n1, n2)
    f["name_partial"] = _cp(fuzz.partial_ratio, n1, n2)
    f["core_ratio"] = _cp(fuzz.ratio, c1, c2)
    f["core_tsort"] = _cp(fuzz.token_sort_ratio, c1, c2)
    f["core_tset"] = _cp(fuzz.token_set_ratio, c1, c2)
    f["core_jw"] = _cp(JaroWinkler.normalized_similarity, c1, c2)
    f["skel_ratio"] = _cp(fuzz.ratio, k1, k2)
    f["skel_tsort"] = _cp(fuzz.token_sort_ratio, k1, k2)
    f["addr_ratio"] = _cp(fuzz.ratio, a1, a2)
    f["addr_tsort"] = _cp(fuzz.token_sort_ratio, a1, a2)
    f["addr_tset"] = _cp(fuzz.token_set_ratio, a1, a2)
    f["addr_partial"] = _cp(fuzz.partial_ratio, a1, a2)
    # concatenations / domains: 'energyprivate' vs 'raj energy private'
    g1 = p["core_n_1"].str.replace_all(" ", "").to_list()
    g2 = p["core_n_2"].str.replace_all(" ", "").to_list()
    # cleaned variants: filler words / junk numbers / PO boxes / URLs / place junk removed
    cc1, cc2 = p["core_c_1"].to_list(), p["core_c_2"].to_list()
    ac1, ac2 = p["addr_c_1"].to_list(), p["addr_c_2"].to_list()
    f["corec_ratio"] = _cp(fuzz.ratio, cc1, cc2)
    f["corec_tsort"] = _cp(fuzz.token_sort_ratio, cc1, cc2)
    f["corec_tset"] = _cp(fuzz.token_set_ratio, cc1, cc2)
    f["addrc_ratio"] = _cp(fuzz.ratio, ac1, ac2)
    f["addrc_tsort"] = _cp(fuzz.token_sort_ratio, ac1, ac2)
    f["addrc_tset"] = _cp(fuzz.token_set_ratio, ac1, ac2)
    f["addrc_partial"] = _cp(fuzz.partial_ratio, ac1, ac2)
    f["glued_partial"] = _cp(fuzz.partial_ratio, g1, g2)
    f["glued_ratio"] = _cp(fuzz.ratio, g1, g2)
    return f


def _token_cover(p, col_a, col_b, thr=80.0):
    """For each pair: #tokens of col_a with no fuzzy match (ratio >= thr) among the tokens of col_b."""
    d = p.select(pl.int_range(pl.len(), dtype=pl.Int64).alias("i"),
                 pl.col(col_a).str.split(" ").alias("ta"), pl.col(col_b).str.split(" ").alias("tb"))
    a = (d.explode("ta").filter(pl.col("ta").is_not_null() & (pl.col("ta") != ""))
         .with_row_index("j").explode("tb").with_columns(pl.col("tb").fill_null("")))
    sc = process.cpdist(a["ta"].to_list(), a["tb"].to_list(), scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    best = a.select("i", "j").with_columns(pl.Series("s", sc)).group_by("j").agg(pl.col("i").first(), pl.col("s").max())
    miss = best.group_by("i").agg((pl.col("s") < thr).sum().alias("miss"), pl.len().alias("n"))
    out = pl.DataFrame({"i": np.arange(p.height, dtype=np.int64)}).join(miss, on="i", how="left").fill_null(0).sort("i")
    return out["miss"].to_numpy().astype(np.float32), out["n"].to_numpy().astype(np.float32)


def token_features(p):
    """Siblings swap a word ('Wright Horizon Equipment' vs '... Petroleum'); true matches add filler words
    ('Group', 'Partners'). Count unmatched core tokens in each direction."""
    m1, n1 = _token_cover(p, "core_n_1", "core_n_2")
    m2, n2 = _token_cover(p, "core_n_2", "core_n_1")
    c1, _ = _token_cover(p, "core_c_1", "core_c_2")
    c2, _ = _token_cover(p, "core_c_2", "core_c_1")
    # address words: exact set difference (a fuzzy all-pairs version is too memory-hungry at 18M pairs)
    at = p.select(pl.col("addr_c_1").str.split(" ").alias("a1"), pl.col("addr_c_2").str.split(" ").alias("a2")).select(
        pl.col("a1").list.set_difference(pl.col("a2")).list.len().alias("m"), pl.col("a1").list.len().alias("n"))
    a1, na1 = at["m"].to_numpy().astype(np.float32), at["n"].to_numpy().astype(np.float32)
    return {"tok_miss_1": m1, "tok_frac_miss_1": m1 / np.maximum(n1, 1),
            "tok_miss_2": m2, "tok_frac_miss_2": m2 / np.maximum(n2, 1), "tok_n_1": n1,
            "tokc_miss_1": c1, "tokc_miss_2": c2, "addr_tok_miss_1": a1,
            "addr_tok_frac_miss_1": a1 / np.maximum(na1, 1)}


def _initials(col):
    return col.str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("")


def rule_features(p):
    """Number and acronym features, vectorised in polars."""
    e = p.lazy().with_columns(
        pl.col("addr_c_1").str.extract_all(r"[0-9]+").alias("num1"),
        pl.col("addr_c_2").str.extract_all(r"[0-9]+").alias("num2"),
        pl.col("legal_1").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("lg1"),
        pl.col("legal_2").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("lg2"),
        _initials(pl.col("core_n_1")).alias("ini1"),
        _initials(pl.col("core_n_2")).alias("ini2"),
        pl.col("core_n_1").str.replace_all(" ", "").alias("g1"),
        pl.col("core_n_2").str.replace_all(" ", "").alias("g2"),
    ).with_columns(
        pl.col("num1").list.first().fill_null("").alias("f1"),
        pl.col("num2").list.first().fill_null("").alias("f2"),
        pl.col("num1").list.len().alias("nn1"),
        pl.col("num2").list.len().alias("nn2"),
        pl.col("num1").list.set_intersection(pl.col("num2")).list.len().alias("ninter"),
        pl.col("num1").list.set_difference(pl.col("num2")).list.len().alias("nonly1"),
        pl.col("num2").list.set_difference(pl.col("num1")).list.len().alias("nonly2"),
        pl.col("num1").list.set_union(pl.col("num2")).list.len().alias("nunion"),
    ).select(
        (pl.col("nn1").cast(pl.Float32)).alias("num_n1"),
        pl.col("nonly1").cast(pl.Float32).alias("num_only_1"),
        pl.col("nonly2").cast(pl.Float32).alias("num_only_2"),
        # house-number distance: siblings sit a few numbers apart (544 vs 557), noise keeps a suffix (476 vs 7476)
        (pl.col("f1").str.slice(0, 9).cast(pl.Int64, strict=False) - pl.col("f2").str.slice(0, 9).cast(pl.Int64, strict=False))
        .abs().log1p().fill_null(-1).cast(pl.Float32).alias("num_first_logdiff"),
        (pl.col("f1").str.slice(-2) == pl.col("f2").str.slice(-2)).fill_null(False).cast(pl.Float32).alias("num_first_last2_eq"),
        (pl.col("nn2").cast(pl.Float32)).alias("num_n2"),
        (pl.col("ninter") / pl.col("nunion").clip(1, None)).cast(pl.Float32).alias("num_jacc"),
        (pl.col("ninter") > 0).cast(pl.Float32).alias("num_any"),
        ((pl.col("nn1") > 0) & (pl.col("nn2") > 0) & (pl.col("ninter") == 0)).cast(pl.Float32).alias("num_conflict"),
        ((pl.col("f1") == pl.col("f2")) & (pl.col("f1") != "")).cast(pl.Float32).alias("num_first_eq"),
        (((pl.col("f1").str.len_chars() > 1) & (pl.col("f2").str.len_chars() > 1)) &
         (pl.col("f1").str.ends_with(pl.col("f2")) | pl.col("f2").str.ends_with(pl.col("f1")))
         ).cast(pl.Float32).alias("num_first_suffix"),
        ((pl.col("ini1") == pl.col("g2")) & (pl.col("ini1").str.len_chars() > 1)).cast(pl.Float32).alias("acro_12"),
        ((pl.col("ini2") == pl.col("g1")) & (pl.col("ini2").str.len_chars() > 1)).cast(pl.Float32).alias("acro_21"),
        (pl.col("g2").str.starts_with(pl.col("ini1")) & (pl.col("ini1").str.len_chars() > 1)).cast(pl.Float32).alias("acro_prefix"),
        (pl.col("addr_n_1") == "").cast(pl.Float32).alias("addr_empty_1"),
        (pl.col("addr_n_2") == "").cast(pl.Float32).alias("addr_empty_2"),
        (pl.col("addr_n_1") == pl.col("addr_n_2")).cast(pl.Float32).alias("addr_exact"),
        (pl.col("addr_c_1") == pl.col("addr_c_2")).cast(pl.Float32).alias("addrc_exact"),
        (pl.col("core_c_1") == pl.col("core_c_2")).cast(pl.Float32).alias("corec_exact"),
        # legal form: 'LLC' vs 'P.C.' separates same-name entities; noise also drops legal words
        pl.col("lg1").list.len().cast(pl.Float32).alias("legal_n1"),
        pl.col("lg2").list.len().cast(pl.Float32).alias("legal_n2"),
        pl.col("lg1").list.set_intersection(pl.col("lg2")).list.len().cast(pl.Float32).alias("legal_inter"),
        ((pl.col("lg1").list.len() > 0) & (pl.col("lg2").list.len() > 0) &
         (pl.col("lg1").list.set_intersection(pl.col("lg2")).list.len() == 0)).cast(pl.Float32).alias("legal_conflict"),
        (pl.col("core_n_1") == pl.col("core_n_2")).cast(pl.Float32).alias("core_exact"),
        (pl.col("skel_1") == pl.col("skel_2")).cast(pl.Float32).alias("skel_exact"),
        pl.col("name_n_1").str.len_chars().cast(pl.Float32).alias("name_len_1"),
        pl.col("name_n_2").str.len_chars().cast(pl.Float32).alias("name_len_2"),
        pl.col("addr_n_2").str.len_chars().cast(pl.Float32).alias("addr_len_2"),
        (pl.col("src_2") == 3).cast(pl.Float32).alias("is_s3"),
    ).collect()
    return {c: e[c].to_numpy() for c in e.columns}


def context_features(cand, zcos):
    """Candidate-set context: sizes, and the record's rank among its entity's candidates by joint cosine.

    Must be computed on the FULL candidate set (as at test time), then joined to the rows being featurised.
    """
    c = cand.select("s1_rid", "rid").with_columns(pl.Series("zc", zcos))
    return c.with_columns(
        pl.len().over("s1_rid").cast(pl.Float32).alias("s1_ncand"),
        pl.len().over("rid").cast(pl.Float32).alias("rec_ncand"),
        pl.col("zc").rank("ordinal", descending=True).over("s1_rid").cast(pl.Float32).alias("s1_rank"),
        (pl.col("zc").max().over("s1_rid") - pl.col("zc")).cast(pl.Float32).alias("s1_gap"),
    ).drop("zc")


def name_frequency(recs):
    """Per record: how many Source-1 entities of its country share its core name / skeleton.
    A record whose exact core name belongs to a single entity is almost surely that entity's."""
    s1 = recs.filter(pl.col("src") == 1)
    fc = s1.group_by("country", "core_n").len("f_core")
    fs = s1.group_by("country", "skel").len("f_skel")
    return (recs.select("rid", "country", "core_n", "skel")
            .join(fc, on=["country", "core_n"], how="left").join(fs, on=["country", "skel"], how="left")
            .select("rid", pl.col("f_core").fill_null(0).cast(pl.Float32), pl.col("f_skel").fill_null(0).cast(pl.Float32)))


COMPETE = ("z_cos", "za_cos", "zn_cos", "core_tset", "core_ratio", "name_tset", "skel_ratio", "addr_tset",
           "addr_ratio", "num_first_eq", "core_exact", "skel_exact", "corec_tset", "addrc_tset", "num_jacc",
           "legal_inter")


def competition_features(df):
    """Each pair versus the record's best OTHER candidate, plus exact-name counts per record and entity."""
    cols = [c for c in COMPETE if c in df.columns]
    top = df.group_by("rid").agg([pl.col(c).top_k(2).alias(f"_t_{c}") for c in cols])
    d = df.select("s1_rid", "rid", *cols).join(top, on="rid", how="left", maintain_order="left")
    exprs = []
    for c in cols:
        t1 = pl.col(f"_t_{c}").list.get(0, null_on_oob=True)
        t2 = pl.col(f"_t_{c}").list.get(1, null_on_oob=True).fill_null(-1.0)
        other = pl.when(pl.col(c) >= t1).then(t2).otherwise(t1)
        exprs.append((pl.col(c) - other).cast(pl.Float32).alias(f"lead_{c}"))
    d = d.with_columns(exprs).with_columns(
        pl.col("core_exact").sum().over("rid").cast(pl.Float32).alias("rec_n_core_exact"),
        pl.col("skel_exact").sum().over("rid").cast(pl.Float32).alias("rec_n_skel_exact"),
    )
    keep = [f"lead_{c}" for c in cols] + ["rec_n_core_exact", "rec_n_skel_exact"]
    return d.select(keep)


def _rid_chunks(rids, chunk):
    """Chunk boundaries on a rid-sorted array that never split one record's candidates (competition
    features compare a record's candidates with each other)."""
    bounds, a = [], 0
    while a < len(rids):
        b = min(a + chunk, len(rids))
        if b < len(rids):
            b = int(np.searchsorted(rids, rids[b - 1], side="right"))
        bounds.append((a, b))
        a = b
    return bounds


def iter_build(cand, recs, nb, embs, full=None, chunk=1_500_000, log=print):
    """Yield complete feature frames (context, embedding, string, rule, token, markers, competition) for
    rid-aligned chunks of the candidate pairs. Memory stays bounded by one chunk."""
    t0 = time.time()
    cand = cand.sort(["rid", "s1_rid"])
    full = cand if full is None else full.select("s1_rid", "rid")
    zfull = _rowdot(embs["z"], embs["z"], full["s1_rid"].to_numpy().astype(np.int64), full["rid"].to_numpy().astype(np.int64))
    ctx_all = context_features(full, zfull)
    del zfull
    # entity-side exact-name count over the FULL candidate set (chunks split entities; train == test)
    core = recs.select(pl.col("rid").cast(pl.Int32), "core_n")
    ex = (full.select(pl.col("s1_rid").cast(pl.Int32), pl.col("rid").cast(pl.Int32))
          .join(core.rename({"rid": "s1_rid", "core_n": "c1"}), on="s1_rid", how="left")
          .join(core.rename({"core_n": "c2"}), on="rid", how="left")
          .group_by("s1_rid").agg((pl.col("c1") == pl.col("c2")).sum().cast(pl.Float32).alias("s1_n_core_exact")))
    ctx_all = ctx_all.with_columns(pl.col("s1_rid").cast(pl.Int32), pl.col("rid").cast(pl.Int32)).join(
        ex, on="s1_rid", how="left").with_columns(pl.col("s1_n_core_exact").fill_null(0))
    del core, ex
    cols = ["rid", "name_n", "core_n", "skel", "addr_n", "src", "core_c", "addr_c", "legal", "f_core", "f_skel"]
    side = (recs.select("rid", "name_n", "core_n", "skel", "addr_n", "src", "core_c", "addr_c", "legal")
            .join(name_frequency(recs), on="rid"))
    markers = recs.select("rid", *[c for c in MARKERS if c in recs.columns])
    side1 = side.rename({k: f"{k}_1" for k in cols if k != "rid"})
    side2 = side.rename({k: f"{k}_2" for k in cols if k != "rid"})
    bounds = _rid_chunks(cand["rid"].to_numpy(), chunk)
    for i, (a, b) in enumerate(bounds):
        c = cand.slice(a, b - a)
        e = embedding_features(c, nb, embs, recs.height)
        p = (c.join(side1, left_on="s1_rid", right_on="rid", how="left", maintain_order="left")
             .join(side2, on="rid", how="left", maintain_order="left"))
        g = {**string_features(p), **rule_features(p), **token_features(p)}
        for k in ("f_core_1", "f_skel_1", "f_core_2", "f_skel_2"):
            g[k] = p[k].to_numpy()
        mk = c.select("rid").join(markers, on="rid", how="left", maintain_order="left")   # record-side priors
        for k in markers.columns[1:]:
            g[k] = mk[k].to_numpy()
        del p
        df = pl.concat([c.join(ctx_all, on=["s1_rid", "rid"], how="left", maintain_order="left"),
                        pl.DataFrame(e), pl.DataFrame(g)], how="horizontal")
        df = pl.concat([df, competition_features(df)], how="horizontal")
        log(f"[features] chunk {i + 1}/{len(bounds)}: {b:,}/{cand.height:,} pairs ({time.time() - t0:.0f}s)")
        yield df


def build(cand, recs, nb, embs, full=None, chunk=1_500_000, log=print):
    """All features in one frame (small inputs: samples, analysis)."""
    return pl.concat(list(iter_build(cand, recs, nb, embs, full=full, chunk=chunk, log=log)))


def build_parts(cand, recs, nb, embs, out_dir, full=None, post=None, chunk=1_500_000, log=print):
    """Write features as parquet parts (bounded memory for tens of millions of pairs)."""
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + "_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for i, df in enumerate(iter_build(cand, recs, nb, embs, full=full, chunk=chunk, log=log)):
        (post(df) if post else df).write_parquet(tmp / f"part-{i:04d}.parquet")
    tmp.rename(out_dir)        # only complete feature sets become visible (resumability)


def read_parts(out_dir, columns=None):
    return pl.read_parquet(Path(out_dir) / "*.parquet", columns=columns)


def feature_columns(df):
    return [c for c in df.columns if c not in ("s1_rid", "rid", "label", "fold")]
