"""Stage 6: from pair probabilities to final match lists, and the challenge metric.

1. One entity per record: each Source-2/3 record keeps only its highest-probability Source-1 entity
   (in the training labels every record matches at most one entity).
2. Per Source-1 entity, choose the prefix of its records (sorted by probability) that maximises the
   expected F0.5, where "no matches" is scored by P(all candidates are wrong). Alternatively a plain
   probability threshold. Both are tuned on out-of-fold predictions.
"""
import numpy as np
import polars as pl


def assign(pred: pl.DataFrame) -> pl.DataFrame:
    """pred: (s1_rid, rid, p). Keep each record's best entity only."""
    return (pred.sort(["rid", "p", "s1_rid"], descending=[False, True, False])
            .unique("rid", keep="first", maintain_order=True))


def select_threshold(assigned: pl.DataFrame, t: float) -> pl.DataFrame:
    return assigned.filter(pl.col("p") >= t).select("s1_rid", "rid")


def select_expected_f(assigned: pl.DataFrame, alpha: float = 1.0, min_p: float = 0.0) -> pl.DataFrame:
    """Per entity, the top-m records maximising E[F0.5] ~ 1.25*sum_top_m(p) / (m + 0.25*sum_all(p))."""
    d = (assigned.with_columns((pl.col("p") ** alpha).clip(1e-6, 1 - 1e-6).alias("q"))
         .sort(["s1_rid", "q"], descending=[False, True])
         .with_columns(
             pl.col("q").cum_sum().over("s1_rid").alias("tp"),
             pl.int_range(1, pl.len() + 1).over("s1_rid").alias("m"),
             pl.col("q").sum().over("s1_rid").alias("g"),
             (1 - pl.col("q")).log().sum().over("s1_rid").exp().alias("ef0"),
         )
         .with_columns((1.25 * pl.col("tp") / (pl.col("m") + 0.25 * pl.col("g"))).alias("ef")))
    best = d.group_by("s1_rid").agg(
        pl.col("m").get(pl.col("ef").arg_max()).alias("m_star"),
        pl.col("ef").max().alias("ef_star"),
        pl.col("ef0").first().alias("ef0"),
    )
    d = d.join(best, on="s1_rid")
    keep = (pl.col("m") <= pl.col("m_star")) & (pl.col("ef_star") > pl.col("ef0")) & (pl.col("q") >= min_p)
    return d.filter(keep).select("s1_rid", "rid")


def macro_f05(pred_pairs: pl.DataFrame, true_pairs: pl.DataFrame, s1_eval: np.ndarray) -> float:
    """Challenge metric over the given Source-1 entities (singletons included)."""
    ev = pl.DataFrame({"s1_rid": s1_eval.astype(np.int32)})
    pp = pred_pairs.select(pl.col("s1_rid").cast(pl.Int32), pl.col("rid").cast(pl.Int32)).join(ev, on="s1_rid")
    tp_ = true_pairs.select(pl.col("s1_rid").cast(pl.Int32), pl.col("rid").cast(pl.Int32)).join(ev, on="s1_rid")
    hit = pp.join(tp_, on=["s1_rid", "rid"], how="semi")
    per = (ev.join(pp.group_by("s1_rid").len("n_pred"), on="s1_rid", how="left")
           .join(tp_.group_by("s1_rid").len("n_true"), on="s1_rid", how="left")
           .join(hit.group_by("s1_rid").len("n_hit"), on="s1_rid", how="left").fill_null(0))
    npred, ntrue, nhit = (per[c].to_numpy().astype(np.float64) for c in ("n_pred", "n_true", "n_hit"))
    prec = np.where(npred > 0, nhit / np.maximum(npred, 1), 0.0)
    rec = np.where(ntrue > 0, nhit / np.maximum(ntrue, 1), 0.0)
    f = np.where(prec + rec > 0, 1.25 * prec * rec / np.maximum(0.25 * prec + rec, 1e-12), 0.0)
    f = np.where((npred == 0) & (ntrue == 0), 1.0, f)
    return float(f.mean())


def tune(pred: pl.DataFrame, true_pairs: pl.DataFrame, s1_eval: np.ndarray, log=print):
    """Grid-search the decision rule on out-of-fold predictions. Returns (best_rule, results)."""
    assigned = assign(pred)
    results = []
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        results.append(({"rule": "threshold", "t": t}, macro_f05(select_threshold(assigned, t), true_pairs, s1_eval)))
    for alpha in (0.7, 1.0, 1.5, 2.0, 3.0):
        for min_p in (0.0, 0.2, 0.4):
            r = {"rule": "expected_f", "alpha": alpha, "min_p": min_p}
            results.append((r, macro_f05(select_expected_f(assigned, alpha, min_p), true_pairs, s1_eval)))
    for r, s in results:
        log(f"[decide] {r} -> F0.5 {s:.5f}")
    best = max(results, key=lambda x: x[1])
    return best[0], results


def apply_rule(pred: pl.DataFrame, rule: dict) -> pl.DataFrame:
    assigned = assign(pred)
    if rule["rule"] == "threshold":
        return select_threshold(assigned, rule["t"])
    return select_expected_f(assigned, rule["alpha"], rule["min_p"])
