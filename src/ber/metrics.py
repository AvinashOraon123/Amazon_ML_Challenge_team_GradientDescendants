"""Evaluation exactly as the leaderboard computes it, plus the decision rule.

Leaderboard metric: F_0.5 computed PER SOURCE-1 ENTITY and macro-averaged over all
S1 entities. An S1 entity with no true matches scores 1.0 for an empty prediction and
0.0 for any prediction; an entity with true matches but an empty prediction scores 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def explode_gt(gt: pd.DataFrame) -> pd.DataFrame:
    """ground-truth TSV -> long frame (source1_entity_id, candidate_entity_id) of true pairs."""
    g = gt.assign(candidate_entity_id=gt["matched_entity_ids"].fillna("").str.split(","))
    g = g.explode("candidate_entity_id")
    g = g[g["candidate_entity_id"].fillna("") != ""]
    return g[["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)


def per_entity_f05(n_true: np.ndarray, n_pred: np.ndarray, n_tp: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """Vectorised per-entity F_beta with the competition's edge-case rules."""
    n_true, n_pred, n_tp = (np.asarray(a, dtype=np.float64) for a in (n_true, n_pred, n_tp))
    b2 = beta**2
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_pred > 0, n_tp / n_pred, 0.0)
        r = np.where(n_true > 0, n_tp / n_true, 0.0)
        f = np.where((p + r) > 0, (1 + b2) * p * r / (b2 * p + r), 0.0)
    both_empty = (n_true == 0) & (n_pred == 0)
    return np.where(both_empty, 1.0, f)


def macro_f05(entity_ids: np.ndarray, true_pairs: pd.DataFrame, pred_pairs: pd.DataFrame,
              beta: float = 0.5) -> dict:
    """Macro F_beta over `entity_ids` (every S1 in the evaluation set, incl. singletons).

    true_pairs / pred_pairs: frames with columns (source1_entity_id, candidate_entity_id).
    """
    ents = pd.Index(pd.unique(entity_ids), name="source1_entity_id")
    tp_keys = true_pairs.merge(pred_pairs, on=["source1_entity_id", "candidate_entity_id"])
    n_true = true_pairs.groupby("source1_entity_id").size().reindex(ents, fill_value=0).to_numpy()
    n_pred = pred_pairs.groupby("source1_entity_id").size().reindex(ents, fill_value=0).to_numpy()
    n_tp = tp_keys.groupby("source1_entity_id").size().reindex(ents, fill_value=0).to_numpy()
    f = per_entity_f05(n_true, n_pred, n_tp, beta)
    tot_p, tot_t = n_pred.sum(), n_true.sum()
    return {
        "macro_f05": float(f.mean()),
        "pair_precision": float(n_tp.sum() / tot_p) if tot_p else 1.0,
        "pair_recall": float(n_tp.sum() / tot_t) if tot_t else 1.0,
        "singleton_acc": float(f[n_true == 0].mean()) if (n_true == 0).any() else float("nan"),
        "n_entities": int(len(ents)),
        "n_pred_pairs": int(tot_p),
    }


def assign_one_to_one(scored: pd.DataFrame, prob_col: str = "prob") -> pd.DataFrame:
    """Keep, for each S2/S3 record, only its highest-probability S1 candidate.

    The training labels show every S2/S3 record is matched to at most one S1 entity, so
    a record linked to two S1s is guaranteed to contain at least one false positive.
    """
    s = scored.sort_values(["candidate_entity_id", prob_col], ascending=[True, False], kind="stable")
    return s.drop_duplicates("candidate_entity_id", keep="first")


def sweep_thresholds(entity_ids: np.ndarray, true_pairs: pd.DataFrame, scored: pd.DataFrame,
                     thresholds=None, prob_col: str = "prob", one_to_one: bool = True,
                     beta: float = 0.5) -> pd.DataFrame:
    """Macro F_beta (and pair P/R) vs threshold, applying the same decision rule used at
    inference: optional one-to-one assignment, then prob >= threshold."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.99, 0.01), 3)
    base = assign_one_to_one(scored, prob_col) if one_to_one else scored
    base = base[["source1_entity_id", "candidate_entity_id", prob_col]]
    tp_flag = base.merge(true_pairs.assign(_y=1), on=["source1_entity_id", "candidate_entity_id"], how="left")["_y"].fillna(0).to_numpy()

    ents = pd.Index(pd.unique(entity_ids))
    n_true = true_pairs.groupby("source1_entity_id").size().reindex(ents, fill_value=0).to_numpy()
    code = pd.Categorical(base["source1_entity_id"], categories=ents).codes
    ok = code >= 0
    code, prob, tp_flag = code[ok], base[prob_col].to_numpy()[ok], tp_flag[ok]
    rows = []
    for t in thresholds:
        m = prob >= t
        n_pred = np.bincount(code[m], minlength=len(ents))
        n_tp = np.bincount(code[m], weights=tp_flag[m], minlength=len(ents))
        f = per_entity_f05(n_true, n_pred, n_tp, beta)
        rows.append({
            "threshold": float(t), "macro_f05": f.mean(),
            "pair_precision": n_tp.sum() / max(1, n_pred.sum()),
            "pair_recall": n_tp.sum() / max(1, n_true.sum()),
            "singleton_acc": f[n_true == 0].mean() if (n_true == 0).any() else np.nan,
        })
    return pd.DataFrame(rows)


def true_counts(true_keys: np.ndarray, n1: int) -> np.ndarray:
    """Number of labelled matches per S1 index (from ALL labels, incl. pairs the candidate
    stage missed — those still count against recall)."""
    return np.bincount((true_keys % n1).astype(np.int64), minlength=n1)


def sweep_idx(ent_idx: np.ndarray, n_true_all: np.ndarray, s1_idx: np.ndarray, label: np.ndarray,
              prob: np.ndarray, thresholds=None, beta: float = 0.5) -> pd.DataFrame:
    """Leaderboard macro F-beta vs threshold on integer ids.

    ent_idx    : S1 indices of the evaluation entities (every one counts, even with no candidates)
    n_true_all : per-S1 number of labelled matches (true_counts)
    s1_idx/label/prob : scored candidate pairs of (at least) those entities
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.995, 0.01), 3)
    ent_idx = np.asarray(ent_idx)
    code = np.full(len(n_true_all), -1, np.int64)
    code[ent_idx] = np.arange(len(ent_idx))
    c = code[s1_idx]
    ok = c >= 0
    c, label, prob = c[ok], label[ok].astype(np.float64), prob[ok]
    n_true = n_true_all[ent_idx]
    rows = []
    for t in thresholds:
        m = prob >= t
        n_pred = np.bincount(c[m], minlength=len(ent_idx))
        n_tp = np.bincount(c[m], weights=label[m], minlength=len(ent_idx))
        f = per_entity_f05(n_true, n_pred, n_tp, beta)
        rows.append({"threshold": float(t), "macro_f05": f.mean(),
                     "pair_precision": n_tp.sum() / max(1, n_pred.sum()),
                     "pair_recall": n_tp.sum() / max(1, n_true.sum()),
                     "singleton_acc": f[n_true == 0].mean() if (n_true == 0).any() else np.nan})
    return pd.DataFrame(rows)


def expected_f_select(s1_idx: np.ndarray, prob: np.ndarray, beta: float = 0.5, alpha: float = 1.0,
                      floor: float = 0.0) -> np.ndarray:
    """Per-entity decision that maximises EXPECTED F-beta instead of using one global threshold.

    For each S1 entity, candidates are sorted by probability and the top-k set is chosen to
    maximise   E[F] ~= (1+b^2) * sum_{i<=k} p_i / (b^2 * N + k),   N = alpha * sum_i p_i
    (expected number of true matches). The empty set is chosen when
    P(no match) ~= prod_i (1 - p_i) is larger — which is exactly the singleton case the
    leaderboard rewards with 1.0. alpha (calibration of N) and floor (min probability to be
    eligible) are tuned on validation data. Returns a boolean mask over the input rows.
    """
    b2 = beta * beta
    n = len(prob)
    if n == 0:
        return np.zeros(0, bool)
    p = np.clip(prob.astype(np.float64), 1e-6, 1 - 1e-6)
    order = np.lexsort((-p, s1_idx))
    s, ps = s1_idx[order], p[order]
    start = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    size = np.diff(np.r_[start, n])
    gid = np.repeat(np.arange(len(start)), size)
    k = np.arange(n) - np.repeat(start, size) + 1                      # rank within entity (1-based)
    cs = np.cumsum(ps)
    cs = cs - np.repeat(np.r_[0.0, cs[start[1:] - 1]], size)           # within-entity cumulative sum
    tot = np.add.reduceat(ps, start)
    N = alpha * tot[gid]
    val = (1 + b2) * cs / (b2 * N + k)
    val[ps < floor] = -np.inf
    best_val = np.maximum.reduceat(val, start)
    best_k = np.zeros(len(start), np.int64)
    # first k attaining the per-entity maximum
    is_best = val == best_val[gid]
    first = np.full(len(start), n, np.int64)
    np.minimum.at(first, gid[is_best], np.flatnonzero(is_best))
    best_k = np.where(first < n, k[np.minimum(first, n - 1)], 0)
    p_none = np.exp(np.add.reduceat(np.log1p(-ps), start))
    best_k = np.where(p_none >= best_val, 0, best_k)
    sel_sorted = k <= best_k[gid]
    sel = np.zeros(n, bool)
    sel[order] = sel_sorted
    return sel


def eval_selection(ent_idx: np.ndarray, n_true_all: np.ndarray, s1_idx: np.ndarray, label: np.ndarray,
                   sel: np.ndarray, beta: float = 0.5) -> dict:
    """Leaderboard macro F-beta of an arbitrary selection mask over scored pairs."""
    code = np.full(len(n_true_all), -1, np.int64)
    code[ent_idx] = np.arange(len(ent_idx))
    c = code[s1_idx]
    m = (c >= 0) & sel
    n_pred = np.bincount(c[m], minlength=len(ent_idx))
    n_tp = np.bincount(c[m], weights=label[m].astype(np.float64), minlength=len(ent_idx))
    n_true = n_true_all[ent_idx]
    f = per_entity_f05(n_true, n_pred, n_tp, beta)
    return {"macro_f05": f.mean(), "pair_precision": n_tp.sum() / max(1, n_pred.sum()),
            "pair_recall": n_tp.sum() / max(1, n_true.sum()),
            "singleton_acc": f[n_true == 0].mean() if (n_true == 0).any() else np.nan}
