"""Stage 3: candidate generation.

Every Source-2/3 record searches the Source-1 entities of its own country with the bi-encoder and
keeps a few nearest ones. Because each record belongs to at most one Source-1 entity, searching in
this direction keeps candidate lists short: a record contributes to at most `k` Source-1 lists, and
only when the other neighbours score close to its best one.

Search runs in three embedding views (joint, address-only, name-only). Policy per view, applied
per record with neighbours sorted by cosine score s_1 >= s_2 >= ...:
    keep rank r  iff  r <= k  and  s_r >= s_1 - margin  and  s_r >= floor
The candidate set is the union over views. The address view rescues records whose name is
unrelated (brand names, acronyms); the name view covers records with no address.
"""
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from . import encoder as E
from . import knn
from .train_encoder import Store, encode_all


VIEWS = ("z", "za", "zn")   # joint, address-only, name-only embeddings


def search(model_path, work_dir, device, k=10, encode_only=False):
    """Encode every record in all views (saved as emb_<view>.npy, fp16) and find the top-k Source-1
    neighbours of every Source-2/3 record in each view (saved as neighbours.npz)."""
    t0 = time.time()
    work_dir = Path(work_dir)
    store = Store(work_dir, device)
    model = E.load(model_path, device)
    embs = encode_all(model, store, np.arange(len(store.src)), parts=VIEWS)
    for v, e in embs.items():
        np.save(work_dir / f"emb_{v}.npy", e.numpy())
    print(f"[block] encoded {len(store.src):,} records ({time.time() - t0:.0f}s)", flush=True)
    if encode_only:
        return None
    s1 = np.nonzero(store.src == 1)[0]
    q = np.nonzero(store.src != 1)[0]
    empty = pl.read_parquet(work_dir / "records.parquet", columns=["addr_n"])["addr_n"].to_numpy()[q] == ""
    arrays = {"q": q}
    for v, e in embs.items():
        # the name view is only used for records without an address: search just those (~3%)
        qi = np.nonzero(empty)[0] if v == "zn" else np.arange(len(q))
        nb_, sc_ = knn.topk_by_group(q[qi], store.country[q[qi]], s1, store.country[s1], e, k, device)
        nbr = np.full((len(q), k), -1, np.int32)
        sc = np.full((len(q), k), -np.inf, np.float16)
        nbr[qi], sc[qi] = nb_, sc_
        arrays[f"nbr_{v}"], arrays[f"sc_{v}"] = nbr, sc
    np.savez(work_dir / "neighbours.npz", **arrays)
    print(f"[block] searched {len(q):,} records in {len(VIEWS)} views ({time.time() - t0:.0f}s)", flush=True)
    return arrays


def load_neighbours(work_dir):
    z = np.load(Path(work_dir) / "neighbours.npz")
    nb = {key: z[key] for key in z.files}
    addr = pl.read_parquet(Path(work_dir) / "records.parquet", columns=["addr_n"])["addr_n"].to_numpy()
    nb["q_empty"] = addr[nb["q"]] == ""      # records without an address (name-only evidence)
    return nb


def _view_pairs(nb, v, k, margin, floor):
    """(query index, S1 row, score) of the neighbours a view keeps under (k, margin, floor).

    View "zn_e" is the name view restricted to records without an address."""
    base = "zn" if v == "zn_e" else v
    nbr, sc = nb[f"nbr_{base}"][:, :k], nb[f"sc_{base}"][:, :k].astype(np.float32)
    keep = (nbr >= 0) & (sc >= sc[:, :1] - margin) & (sc >= floor)
    if v == "zn_e":
        keep &= nb["q_empty"][:, None]
    qi, cc = np.nonzero(keep)
    return qi, nbr[qi, cc].astype(np.int64), sc[qi, cc]


def _cap_mask(s1, score, cap):
    """True for pairs ranked within the top `cap` scores of their Source-1 entity (hub control)."""
    if not cap:
        return np.ones(len(s1), bool)
    order = np.lexsort((-score, s1))
    s1o = s1[order]
    start = np.r_[0, np.nonzero(np.diff(s1o))[0] + 1]
    rank = np.arange(len(s1o)) - np.repeat(start, np.diff(np.r_[start, len(s1o)]))
    m = np.empty(len(s1), bool)
    m[order] = rank < cap
    return m


def policy_pairs(nb, policy):
    """Candidate pairs (query index, S1 row) of a policy: union over views, each view capped per entity.

    policy: {view: (k, margin, floor), ..., optional "cap": max candidates per entity per view}.
    """
    cap = policy.get("cap")
    qs, ss = [], []
    for v, prm in policy.items():
        if v == "cap":
            continue
        qi, s1, sc = _view_pairs(nb, v, *prm)
        m = _cap_mask(s1, sc, cap)
        qs.append(qi[m])
        ss.append(s1[m])
    qi, s1 = np.concatenate(qs), np.concatenate(ss)
    key = np.unique(qi.astype(np.int64) * (1 << 32) + s1)
    return key >> 32, key & ((1 << 32) - 1)


def apply_policy(nb, policy):
    """Candidate frame (s1_rid, rid), one row per pair."""
    qi, s1 = policy_pairs(nb, policy)
    return pl.DataFrame({"s1_rid": s1.astype(np.int32), "rid": nb["q"][qi].astype(np.int32)})


def f05_ceiling(recall):
    """Best per-entity F0.5 a perfect matcher could reach given the fraction of its true matches in the candidates."""
    return np.where(recall > 0, 1.25 * recall / (0.25 + recall), 0.0)


def evaluate_policy(store, cand, holdout_only=True):
    s1_all = np.nonzero(store.src == 1)[0]
    s1_eval = s1_all[store.holdout[s1_all]] if holdout_only else s1_all
    ev = pl.DataFrame({"s1_rid": s1_eval.astype(np.int32)})
    pairs = store.pairs.join(ev, on="s1_rid")
    hit = pairs.join(cand.select("s1_rid", "rid"), on=["s1_rid", "rid"], how="semi")
    per = (
        ev.join(pairs.group_by("s1_rid").len("n_true"), on="s1_rid", how="left")
        .join(hit.group_by("s1_rid").len("n_hit"), on="s1_rid", how="left")
        .join(cand.group_by("s1_rid").len("n_cand"), on="s1_rid", how="left")
        .fill_null(0)
    )
    n_true, n_hit = per["n_true"].to_numpy(), per["n_hit"].to_numpy()
    rec = np.where(n_true > 0, n_hit / np.maximum(n_true, 1), 1.0)
    ceil = np.where(n_true > 0, f05_ceiling(rec), 1.0)       # singletons: the matcher can still predict empty
    return {
        "pair_recall": round(float(n_hit.sum() / max(n_true.sum(), 1)), 5),
        "f05_ceiling": round(float(ceil.mean()), 5),
        "cand_per_s1": round(float(per["n_cand"].mean()), 3),
        "cand_per_s1_all": round(float(len(cand) / len(s1_all)), 3),
        "cand_p99": float(np.quantile(per["n_cand"].to_numpy(), 0.99)),
        "max_cand": int(per["n_cand"].max()),
    }


DEFAULT_GRID = [
    {k: v for k, v in (("z", (kz, mz, -1.0)), ("za", za), ("zn", zn), ("zn_e", zne)) if v is not None}
    for kz in (2, 3, 5) for mz in (0.05, 0.1, 0.2)
    for za in (None, (1, 0.0, 0.8), (2, 0.05, 0.8), (3, 0.05, 0.7))
    for zn in (None,)
    for zne in (None, (3, 0.05, -1.0), (5, 0.1, -1.0))
]
CAPS = (15, 20, 30, 50, 100, None)


def fast_eval(nb, policy, pos, hold_mask, n_true):
    """Vectorised policy statistics on held-out entities.

    pos[i]: true S1 row of query i (-1 if none); hold_mask[r]: row r is a held-out S1; n_true[r]: its #matches.
    """
    qi, s1 = policy_pairs(nb, policy)
    ncand = np.bincount(s1, minlength=len(hold_mask))
    hit = s1 == pos[qi]
    nhit = np.bincount(s1[hit], minlength=len(hold_mask))
    h = np.nonzero(hold_mask)[0]
    nt, nh = n_true[h], nhit[h]
    rec = np.where(nt > 0, nh / np.maximum(nt, 1), 1.0)
    ceil = np.where(nt > 0, f05_ceiling(rec), 1.0)
    return {
        "pair_recall": round(float(nh.sum() / max(nt.sum(), 1)), 5),
        "f05_ceiling": round(float(ceil.mean()), 5),
        "cand_per_s1": round(float(ncand[h].mean()), 3),
        "cand_per_s1_all": round(float(len(s1) / np.count_nonzero(n_true >= 0)), 3),
        "cand_p99": float(np.quantile(ncand[h], 0.99)),
        "max_cand": int(ncand[h].max()),
    }


def sweep(work_dir, grid=None, tol=0.0005, cap_tol=0.0002):
    """Evaluate candidate policies on the held-out entities, then pick the smallest set whose F0.5
    recall ceiling is within `tol` of the best, and the smallest per-entity cap costing < `cap_tol`."""
    store = Store(work_dir, "cpu")
    nb = load_neighbours(work_dir)
    n = len(store.src)
    pos = np.full(n, -1, np.int64)
    pr = store.pairs.to_numpy()
    pos[pr[:, 1]] = pr[:, 0]
    pos = pos[nb["q"]]
    hold_mask = store.holdout & (store.src == 1)
    n_true = np.bincount(pr[:, 0], minlength=n)
    n_true = np.where(store.src == 1, n_true, -1)            # -1 marks non-S1 rows
    rows = []

    def run(policy):
        r = fast_eval(nb, policy, pos, hold_mask, n_true)
        r["policy"] = json.dumps(policy)
        rows.append(r)
        print(json.dumps(r), flush=True)
        return r

    base = [run(p) for p in (grid or DEFAULT_GRID)]
    best = max(r["f05_ceiling"] for r in base)
    chosen = min((r for r in base if r["f05_ceiling"] >= best - tol), key=lambda r: r["cand_per_s1"])
    policy = json.loads(chosen["policy"])
    capped = [run({**policy, "cap": c}) for c in CAPS if c]
    ok = [r for r in capped if r["f05_ceiling"] >= chosen["f05_ceiling"] - cap_tol]
    final = min(ok, key=lambda r: r["cand_per_s1"]) if ok else chosen
    out = pl.DataFrame(rows)
    out.write_csv(Path(work_dir) / "blocking_sweep.csv")
    return json.loads(final["policy"]), final
