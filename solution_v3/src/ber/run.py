"""End-to-end driver. Every stage writes its outputs and is skipped when they already exist, so a
re-run after an interruption resumes from the last completed stage (the encoder additionally resumes
from its last finished epoch).

Layout:
  <work>/train, <work>/test     prepared records, byte matrices, embeddings, neighbours
  <models>/encoder.pt           bi-encoder
  <models>/matcher/             LightGBM + MLP fold models, decision.json
  <out>/                        matching_results.tsv, candidate_pairs.tsv, metrics.json
"""
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from . import blocking, decide, features, matcher
from .prepare import prepare
from .train_encoder import Store, train as train_encoder

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def choose_policy(sweep: pl.DataFrame, tol=0.0005):
    """Smallest candidate set whose F0.5 recall ceiling is within `tol` of the best ceiling in the sweep."""
    best = sweep["f05_ceiling"].max()
    ok = sweep.filter(pl.col("f05_ceiling") >= best - tol).sort("cand_per_s1")
    row = ok.row(0, named=True)
    return json.loads(row["policy"]), row


def _stage_prepare(data, work, split, sample=None):
    if not (Path(work) / split / "records.parquet").exists():
        prepare(data, work, split, sample)
    else:
        log(f"prepare {split}: exists, skipped")


def _stage_search(work, split, model_path, device):
    d = Path(work) / split
    if not (d / "neighbours.npz").exists():
        blocking.search(model_path, d, device, k=10)
    elif not all((d / f"emb_{v}.npy").exists() for v in blocking.VIEWS):
        blocking.search(model_path, d, device, k=10, encode_only=True)   # neighbours reused from an artifact
    else:
        log(f"search {split}: exists, skipped")


def run_all(data, work, models, out, device, epochs=4, mine_from=2, sample=None, batch_size=4096, buckets_log2=22):
    work, models, out = Path(work), Path(models), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    def save_metrics():
        metrics_path.write_text(json.dumps(metrics, indent=1))

    # 1-2. prepare train, train encoder (resumable per epoch)
    _stage_prepare(data, work, "train", sample)
    enc = models / "encoder.pt"
    done = models / "encoder_log.json"
    if not (enc.exists() and done.exists() and len(json.loads(done.read_text())) >= epochs):
        train_encoder(work / "train", models, device=device, epochs=epochs, mine_from=mine_from,
                      batch_size=batch_size, buckets_log2=buckets_log2)
    metrics["encoder_eval"] = json.loads(done.read_text())[-1]
    save_metrics()

    # 3. candidate generation on train + policy choice
    _stage_search(work, "train", enc, device)
    pol_path = models / "policy.json"
    if not pol_path.exists():
        policy, row = blocking.sweep(work / "train")
        pol_path.write_text(json.dumps({"policy": policy, "stats": row}, indent=1))
    pol = json.loads(pol_path.read_text())
    policy = {k: (v if k == "cap" else tuple(v)) for k, v in pol["policy"].items()}
    metrics["blocking_train_holdout"] = pol["stats"]
    save_metrics()
    log(f"candidate policy: {policy} -> {pol['stats']}")

    # 4. matcher features on the validation holdout (written in parts: bounded memory)
    meta = pl.read_parquet(work / "train" / "records.parquet", columns=["rid", "src", "country", "holdout", "entity_id"])
    src, hold_mask, country = meta["src"].to_numpy(), meta["holdout"].to_numpy(), meta["country"].to_numpy()
    s1_all = np.nonzero(src == 1)[0]
    s1_hold = s1_all[hold_mask[s1_all]]
    true_pairs = pl.read_parquet(work / "train" / "pairs.parquet")
    fdir = work / "train" / "features"
    if not fdir.exists():
        recs = pl.read_parquet(work / "train" / "records.parquet")
        nb = blocking.load_neighbours(work / "train")
        full = blocking.apply_policy(nb, policy)
        hold_ids = pl.DataFrame({"s1_rid": s1_hold.astype(np.int32)})
        touched = full.join(hold_ids, on="s1_rid").select("rid").unique()
        cand = full.join(touched, on="rid")                       # every candidate of records near holdout entities
        log(f"train features: {cand.height:,} pairs")
        lab = true_pairs.select(pl.col("s1_rid").cast(pl.Int32), pl.col("rid").cast(pl.Int32)).with_columns(
            pl.lit(1, pl.Float32).alias("label"))

        def post(df):
            return (df.join(lab, on=["s1_rid", "rid"], how="left", maintain_order="left")
                    .with_columns(pl.col("label").fill_null(0),
                                  ((pl.col("rid").cast(pl.Int64) * 2654435761) % 1000 % matcher.NFOLD).cast(pl.Int8).alias("fold"),
                                  pl.col("s1_rid").is_in(hold_ids["s1_rid"].implode()).alias("is_hold")))

        features.build_parts(cand, recs, nb, features.load_embeddings(work / "train"), fdir, full=full, post=post, log=log)
        del recs, nb, full, cand
    feat = features.read_parts(fdir)
    log(f"train features loaded: {feat.shape}")

    # 5-6. matcher CV + decision tuning (OOF), diagnostics
    mdir = models / "matcher"
    matcher.free_gpu()        # the encoder / search stages leave PyTorch's cached GPU memory behind
    cfg, oof_pred = matcher.fit_and_tune(feat, true_pairs, s1_hold, mdir, device, log=log)
    metrics["validation_f05"] = cfg["oof_f05"]
    metrics["decision"] = {"w_lgb": cfg["w_lgb"], "stage2": cfg["stage2"], "rule": cfg["rule"],
                           "stage1_f05": cfg["stage1_f05"]}
    chosen = decide.apply_rule(oof_pred, cfg["rule"])
    per_c = {}
    for c in np.unique(country[s1_hold]):
        per_c[str(c)] = decide.macro_f05(chosen, true_pairs, s1_hold[country[s1_hold] == c])
    metrics["validation_f05_by_country"] = per_c
    n_true = true_pairs.join(pl.DataFrame({"s1_rid": s1_hold.astype(np.int32)}), on="s1_rid").group_by("s1_rid").len()
    single = np.setdiff1d(s1_hold, n_true["s1_rid"].to_numpy())
    metrics["validation_f05_singletons"] = decide.macro_f05(chosen, true_pairs, single)
    metrics["validation_n_entities"] = int(len(s1_hold))
    # comparable with earlier runs: the original 10% holdout (hundreds digit 7) is always part of the holdout
    orig = meta.select("rid", "entity_id").filter(pl.col("rid").is_in(s1_hold.tolist()))
    orig10 = orig.filter((pl.col("entity_id").str.slice(3).cast(pl.Int64) // 100) % 10 == 7)["rid"].to_numpy()
    metrics["validation_f05_orig10"] = decide.macro_f05(chosen, true_pairs, orig10)
    log(f"validation F0.5 on the original 10% holdout: {metrics['validation_f05_orig10']:.5f}")
    save_metrics()
    log(f"VALIDATION macro F0.5 = {cfg['oof_f05']:.5f}  by country {per_c}")
    if "country_transfer" not in metrics:
        sub_ = feat.filter(pl.col("rid") % 4 == 0) if feat.height > 4_000_000 else feat   # whole records kept
        metrics["country_transfer"] = matcher.country_transfer(sub_, country, true_pairs, s1_hold, log=log)
        save_metrics()
        del sub_
    del feat, meta, oof_pred

    # 7. test: prepare, search, candidates, features, predict, decide, write
    _stage_prepare(data, work, "test", sample)
    _stage_search(work, "test", enc, device)
    tfdir = work / "test" / "features"
    if not tfdir.exists():
        trecs = pl.read_parquet(work / "test" / "records.parquet")
        nb = blocking.load_neighbours(work / "test")
        cand = blocking.apply_policy(nb, policy)
        log(f"test features: {cand.height:,} pairs")
        features.build_parts(cand, trecs, nb, features.load_embeddings(work / "test"), tfdir, log=log)
        del trecs, nb, cand
    cols = json.loads((mdir / "feature_columns.json").read_text())
    keep = ["s1_rid", "rid"] + [c for c in matcher.STAGE2_BASE]
    matcher.free_gpu()        # the test search just filled PyTorch's GPU cache; XGBoost predicts on the GPU
    log("predicting test pairs (part by part)")
    parts = []
    for fp in sorted(tfdir.glob("*.parquet")):
        part = pl.read_parquet(fp)
        ps = matcher.predict(part.select(cols).to_numpy(), mdir, device, cfg["w_lgb"])
        parts.append(part.select([c for c in keep if c in part.columns]).with_columns(
            pl.Series("p1", matcher.blend(ps, cfg["w_lgb"]))))
        del part
    tfeat = pl.concat(parts)
    del parts
    log("stage-1 predictions done")
    p_test = tfeat["p1"].to_numpy()
    if cfg.get("stage2"):
        p_test = matcher.predict_stage2(matcher.stack_features(tfeat, p_test), mdir)
    pred = tfeat.select("s1_rid", "rid").with_columns(pl.Series("p", p_test))
    log("stage-2 predictions done")
    matches = decide.apply_rule(pred, cfg["rule"])
    trecs = pl.read_parquet(work / "test" / "records.parquet", columns=["rid", "src", "entity_id", "country"])
    write_outputs(trecs, tfeat.select("s1_rid", "rid"), matches, out)
    s1_test = trecs.filter(pl.col("src") == 1)
    metrics["test"] = {
        "s1_entities": s1_test.height,
        "candidate_pairs": tfeat.height,
        "candidates_per_s1": round(tfeat.height / s1_test.height, 3),
        "matched_pairs": matches.height,
        "matches_per_s1": round(matches.height / s1_test.height, 3),
        "empty_rows": int(s1_test.height - matches["s1_rid"].n_unique()),
        "by_country": score_distribution(trecs, pred, matches),
    }
    save_metrics()
    log(f"done: {json.dumps(metrics['test'])}")
    return metrics


def score_distribution(recs, pred, matches):
    """Per-country test statistics (a sanity check for the unseen country)."""
    c = recs.select(pl.col("rid").cast(pl.Int32).alias("s1_rid"), "country")
    s1 = recs.filter(pl.col("src") == 1).group_by("country").len("n_s1")
    top = (decide.assign(pred).join(c, on="s1_rid").group_by("country")
           .agg(pl.col("p").mean().alias("mean_best_p"), (pl.col("p") > 0.5).mean().alias("share_p_gt_0.5")))
    m = matches.join(c, on="s1_rid").group_by("country").len("n_matches")
    t = s1.join(top, on="country", how="left").join(m, on="country", how="left").with_columns(
        (pl.col("n_matches") / pl.col("n_s1")).alias("matches_per_s1"))
    return {r["country"]: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "country"}
            for r in t.to_dicts()}


def write_outputs(recs, cand, matches, out):
    ids = recs.select(pl.col("rid").cast(pl.Int32), "entity_id")
    s1 = recs.filter(pl.col("src") == 1).select(pl.col("rid").cast(pl.Int32).alias("s1_rid"),
                                                 pl.col("entity_id").alias("source1_entity_id"))

    def lists(pairs, name):
        g = (pairs.select(pl.col("s1_rid").cast(pl.Int32), pl.col("rid").cast(pl.Int32)).unique()
             .join(ids, on="rid").sort("entity_id").group_by("s1_rid")
             .agg(pl.col("entity_id").str.join(",").alias(name)))
        return (s1.join(g, on="s1_rid", how="left").with_columns(pl.col(name).fill_null(""))
                .select("source1_entity_id", name))

    lists(matches, "matched_entity_ids").write_csv(Path(out) / "matching_results.tsv", separator="\t", quote_style="never")
    lists(cand, "candidate_entity_ids").write_csv(Path(out) / "candidate_pairs.tsv", separator="\t", quote_style="never")
    log(f"wrote {out}/matching_results.tsv and candidate_pairs.tsv")


def analyze(data, work, models, out, device, n_show=25000):
    """Train-side stages only (reusing saved models), then export error samples and reusable artifacts."""
    import shutil
    work, models, out = Path(work), Path(models), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _stage_prepare(data, work, "train")
    enc = models / "encoder.pt"
    _stage_search(work, "train", enc, device)
    pol = json.loads((models / "policy.json").read_text())
    policy = {k: (v if k == "cap" else tuple(v)) for k, v in pol["policy"].items()}
    recs = pl.read_parquet(work / "train" / "records.parquet")
    store = Store(work / "train", "cpu")
    s1_all = np.nonzero(store.src == 1)[0]
    s1_hold = s1_all[store.holdout[s1_all]]
    true_pairs = store.pairs
    fpath = work / "train" / "features.parquet"
    if not fpath.exists():
        nb = blocking.load_neighbours(work / "train")
        full = blocking.apply_policy(nb, policy)
        hold_ids = pl.DataFrame({"s1_rid": s1_hold.astype(np.int32)})
        cand = full.join(full.join(hold_ids, on="s1_rid").select("rid").unique(), on="rid")
        feat = features.build(cand, recs, nb, features.load_embeddings(work / "train"), full=full, log=log)
        feat = (feat.join(true_pairs.with_columns(pl.lit(1, pl.Float32).alias("label")), on=["s1_rid", "rid"], how="left")
                .with_columns(pl.col("label").fill_null(0),
                              ((pl.col("rid").cast(pl.Int64) * 2654435761) % 1000 % matcher.NFOLD).cast(pl.Int8).alias("fold"),
                              pl.col("s1_rid").is_in(hold_ids["s1_rid"].implode()).alias("is_hold")))
        feat.write_parquet(fpath)
    feat = pl.read_parquet(fpath)
    cfg, pred = matcher.fit_and_tune(feat, true_pairs, s1_hold, models / "matcher", device, log=log)
    log(f"analysis OOF F0.5 {cfg['oof_f05']:.5f}")
    chosen = decide.apply_rule(pred, cfg["rule"])
    hold_ids = pl.DataFrame({"s1_rid": s1_hold.astype(np.int32)})
    tp = true_pairs.select(pl.col("s1_rid").cast(pl.Int32), pl.col("rid").cast(pl.Int32)).join(hold_ids, on="s1_rid")
    ch = chosen.select(pl.col("s1_rid").cast(pl.Int32), pl.col("rid").cast(pl.Int32)).join(hold_ids, on="s1_rid")
    fp = ch.join(tp, on=["s1_rid", "rid"], how="anti")
    fn = tp.join(ch, on=["s1_rid", "rid"], how="anti")
    owner = true_pairs.select(pl.col("rid").cast(pl.Int32), pl.col("s1_rid").cast(pl.Int32).alias("true_s1"))
    n_true = true_pairs.group_by("s1_rid").len("n_true").with_columns(pl.col("s1_rid").cast(pl.Int32))
    txt = recs.select(pl.col("rid").cast(pl.Int32), "name", "addr", "country")
    keyf = ["z_cos", "za_cos", "zn_cos", "core_tset", "addr_tset", "num_first_eq", "num_first_logdiff", "tok_miss_1",
            "tok_miss_2", "addr_empty_2", "s1_ncand", "rec_ncand", "core_exact", "skel_exact", "f_core_1", "f_core_2",
            "f_skel_2", "rec_n_core_exact", "s1_n_core_exact", "lead_core_tset", "lead_z_cos", "lead_zn_cos", "name_ratio"]
    fsel = feat.select(pl.col("s1_rid").cast(pl.Int32), pl.col("rid").cast(pl.Int32), *[c for c in keyf if c in feat.columns])
    p = pred.select(pl.col("s1_rid").cast(pl.Int32), pl.col("rid").cast(pl.Int32), "p")
    # full prediction table for the held-out-touching records (small: ids, label, p, a few features)
    (feat.select("s1_rid", "rid", "label", "is_hold", *[c for c in keyf if c in feat.columns])
     .with_columns(pl.Series("p", pred["p"].to_numpy())).write_parquet(out / "oof_pred.parquet"))

    def enrich(d):
        d = (d.join(owner, on="rid", how="left").join(n_true, on="s1_rid", how="left").fill_null(0)
             .join(p, on=["s1_rid", "rid"], how="left").join(fsel, on=["s1_rid", "rid"], how="left")
             .join(txt.rename({"rid": "s1_rid", "name": "s1_name", "addr": "s1_addr"}), on="s1_rid", how="left")
             .join(txt.drop("country").rename({"name": "rec_name", "addr": "rec_addr"}), on="rid", how="left"))
        return d

    fpe, fne = enrich(fp), enrich(fn)
    fne = fne.with_columns(pl.col("p").is_not_null().alias("in_candidates"))
    # where did the missed record go instead?
    got = chosen.select(pl.col("rid").cast(pl.Int32), pl.col("s1_rid").cast(pl.Int32).alias("assigned_to"))
    fne = fne.join(got, on="rid", how="left")
    stats = {
        "oof_f05": cfg["oof_f05"], "n_fp": fp.height, "n_fn": fn.height, "n_true": tp.height, "n_pred": ch.height,
        "fp_rec_owned_by_other_s1": float((fpe["true_s1"] > 0).mean()),
        "fp_rec_unmatched_distractor": float((fpe["true_s1"] == 0).mean()),
        "fp_on_singleton_s1": float((fpe["n_true"] == 0).mean()),
        "fn_in_candidates": float(fne["in_candidates"].mean()),
        "fn_assigned_elsewhere": float(fne["assigned_to"].is_not_null().mean()),
        "fn_rec_addr_empty": float((fne["rec_addr"] == "").mean()),
        "fp_by_country": fpe.group_by("country").len().rows(),
        "fn_by_country": fne.group_by("country").len().rows(),
    }
    (out / "error_stats.json").write_text(json.dumps(stats, indent=1))
    fpe.sample(min(n_show, fpe.height), seed=0).write_csv(out / "errors_fp.tsv", separator="\t")
    fne.sample(min(n_show, fne.height), seed=0).write_csv(out / "errors_fn.tsv", separator="\t")
    log(f"error stats: {json.dumps(stats)}")
    art = out / "artifacts" / "train"
    art.mkdir(parents=True, exist_ok=True)
    for f in ("neighbours.npz", "features.parquet"):
        shutil.copy(work / "train" / f, art / f)
    log("saved artifacts")
