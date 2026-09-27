"""Stage 2: train the bi-encoder with contrastive learning, mined hard negatives and held-out evaluation.

Loss per batch of B (Source-1, matched record) pairs:
  * record -> S1 InfoNCE over the batch's S1 rows plus one mined hard-negative S1 per pair
  * S1 -> record InfoNCE over the batch's records
  * 0.5 x the same record -> S1 loss on the name-only and address-only embeddings
Rows sharing the target's Source-1 id are masked, so another match of the same entity is never
treated as a negative.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from . import encoder as E
from . import knn

RECALL_KS = (1, 2, 3, 5, 10, 20)


class Store:
    """Byte matrices for every record (on the GPU when it fits) plus metadata."""

    def __init__(self, work, device):
        work = Path(work)
        self.recs = pl.read_parquet(work / "records.parquet", columns=["rid", "src", "country", "holdout"])
        self.pairs = pl.read_parquet(work / "pairs.parquet") if (work / "pairs.parquet").exists() else None
        self.mats = [torch.from_numpy(np.load(work / f"bytes_{f}.npy")) for f in E.FIELDS]
        if device.startswith("cuda"):
            self.mats = [m.to(device) for m in self.mats]
        self.device = device
        self.src = self.recs["src"].to_numpy()
        self.country = self.recs["country"].to_numpy()
        self.holdout = self.recs["holdout"].to_numpy()

    def batch(self, rows):
        r = torch.as_tensor(rows, device=self.mats[0].device)
        return [m[r].to(self.device, non_blocking=True) for m in self.mats]


def _masked_ce(logits, key_ids, tgt_ids):
    """CE where target of row i is column i; columns whose id equals the row's target id (other than i) are masked."""
    same = key_ids[None, :] == tgt_ids[:, None]
    same.fill_diagonal_(False)
    logits = logits.masked_fill(same, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def encode_all(model, store, rows, parts=("z",), batch=32768):
    out = {p: [] for p in parts}
    model.eval()
    with torch.no_grad():
        for a in range(0, len(rows), batch):
            args = store.batch(rows[a:a + batch])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=store.device.startswith("cuda")):
                z, zn, za, _ = model(*args)
            got = {"z": z, "zn": zn, "za": za}
            for p in parts:
                out[p].append(got[p].half().cpu())
    model.train()
    return {p: torch.cat(v) for p, v in out.items()}


def search_rows(model, store, q_rows, key_rows, k, device, parts=("z",)):
    """Encode only the rows needed, then per-country top-k for each embedding view.

    Returns {view: (nbr global row ids (Nq, k) with -1 padding, scores (Nq, k))}.
    """
    need = np.unique(np.concatenate([key_rows, q_rows]))
    embs = encode_all(model, store, need, parts)
    qp, kp = np.searchsorted(need, q_rows), np.searchsorted(need, key_rows)
    out = {}
    for view, emb in embs.items():
        nbr, sc = knn.topk_by_group(qp, store.country[q_rows], kp, store.country[key_rows], emb, k, device)
        out[view] = (np.where(nbr >= 0, need[np.maximum(nbr, 0)], -1), sc)
    return out


def _ranks(nbr, target):
    hit = nbr == target[:, None]
    return np.where(hit.any(1), hit.argmax(1) + 1, 10 ** 9)


def evaluate(model, store, device, k=20):
    """Recall@k of the true S1 for every held-out pair, searching ALL S1 of the same country (as at test time).

    Views: z (joint), za (address-only), zn (name-only); 'U@k' = true S1 within the top-k of z OR za OR zn.
    """
    s1_rows = np.nonzero(store.src == 1)[0]
    ev = s1_rows[store.holdout[s1_rows]]
    if len(ev) > 300_000:          # large holdouts: a fixed subsample keeps each epoch's evaluation cheap
        ev = ev[ev % 5 == 0]
    pairs = store.pairs.join(pl.DataFrame({"s1_rid": ev.astype(np.int32)}), on="s1_rid")
    q_rows = pairs["rid"].to_numpy().astype(np.int64)
    target = pairs["s1_rid"].to_numpy()
    got = search_rows(model, store, q_rows, s1_rows, k, device, parts=("z", "za", "zn"))
    ranks = {v: _ranks(nbr, target) for v, (nbr, _) in got.items()}
    r = ranks["z"]
    res = {f"R@{kk}": float((r <= kk).mean()) for kk in RECALL_KS if kk <= k}
    res["MRR"] = float(np.where(r < 10 ** 9, 1.0 / r, 0).mean())
    for v in ("za", "zn"):
        for kk in (1, 5, 20):
            res[f"{v}_R@{kk}"] = float((ranks[v] <= kk).mean())
    best = np.minimum(np.minimum(ranks["z"], ranks["za"]), ranks["zn"])
    for kk in (1, 3, 5, 10, 20):
        res[f"U@{kk}"] = float((best <= kk).mean())
    res["n"] = int(len(q_rows))
    for s in (2, 3):
        m = store.src[q_rows] == s
        res[f"R@1_s{s}"] = float((r[m] <= 1).mean())
    return {kk: round(v, 5) if isinstance(v, float) else v for kk, v in res.items()}


def mine_hard_negatives(model, store, train_pairs, device, k=8, rng=None):
    """For each training pair, the highest-ranked WRONG Source-1 (sampled among the top 3 wrong ones)."""
    s1_rows = np.nonzero((store.src == 1) & ~store.holdout)[0]
    q_rows = np.unique(train_pairs[:, 1])
    nbr, _ = search_rows(model, store, q_rows, s1_rows, k, device)["z"]
    pos = np.full(len(store.src), -1, np.int64)
    pos[train_pairs[:, 1]] = train_pairs[:, 0]           # each record has at most one true S1
    wrong = (nbr != pos[q_rows][:, None]) & (nbr >= 0)
    rng = rng or np.random.default_rng(0)
    # pick uniformly among the first 3 wrong neighbours
    order = np.cumsum(wrong, 1)
    pick = rng.integers(1, 4, size=len(q_rows))
    sel = wrong & (order == np.minimum(pick, order[:, -1])[:, None])
    hn_q = np.where(sel.any(1), nbr[np.arange(len(q_rows)), sel.argmax(1)], -1)
    hn = np.full(len(store.src), -1, np.int64)
    hn[q_rows] = hn_q
    return hn[train_pairs[:, 1]]


def make_batches(train_pairs, country, bs, rng):
    """Country-homogeneous batches (harder in-batch negatives), shuffled batch order."""
    key = country[train_pairs[:, 0]]
    batches = []
    for c in np.unique(key):
        idx = np.nonzero(key == c)[0]
        rng.shuffle(idx)
        batches += [idx[a:a + bs] for a in range(0, len(idx), bs) if len(idx[a:a + bs]) >= bs // 4]
    rng.shuffle(batches)
    return batches


def train(work_dir, out_dir, device="cuda", epochs=4, mine_from=2, batch_size=4096, lr_sparse=3e-3,
          lr_dense=1e-3, buckets_log2=22, log_every=200, max_steps=None, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = Store(work_dir, device)
    cfg = E.EncoderConfig(buckets_log2=buckets_log2)
    model = E.Encoder(cfg).to(device)
    opt_s = torch.optim.SparseAdam(model.sparse_parameters(), lr=lr_sparse)
    opt_d = torch.optim.AdamW(model.dense_parameters(), lr=lr_dense, weight_decay=1e-4)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hold = store.holdout
    tp = store.pairs.to_numpy().astype(np.int64)
    tp = tp[~hold[tp[:, 0]]]                                # train only on non-holdout Source-1 entities
    steps_per_epoch = math.ceil(len(tp) / batch_size)
    total = max_steps or steps_per_epoch * epochs
    print(f"[encoder] {len(tp):,} training pairs, {steps_per_epoch} steps/epoch, {total} steps", flush=True)
    hn = None
    log = []
    step, t0, start_ep = 0, time.time(), 0
    ck_path = out / "encoder_ckpt.pt"
    if ck_path.exists():                                     # resume after an interrupted run
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt_s.load_state_dict(ck["opt_s"])
        opt_d.load_state_dict(ck["opt_d"])
        scaler.load_state_dict(ck["scaler"])
        rng.bit_generator.state = ck["rng"]
        step, start_ep, log = ck["step"], ck["epoch"] + 1, ck["log"]
        print(f"[encoder] resumed from epoch {ck['epoch']} (step {step})", flush=True)
    for ep in range(start_ep, epochs):
        if ep >= mine_from:
            tm = time.time()
            hn = mine_hard_negatives(model, store, tp, device, rng=rng)
            print(f"[encoder] mined hard negatives for {np.mean(hn >= 0):.1%} of pairs ({time.time() - tm:.0f}s)", flush=True)
        for bi in make_batches(tp, store.country, batch_size, rng):
            frac = step / total
            for o, base in ((opt_s, lr_sparse), (opt_d, lr_dense)):     # warmup + cosine decay
                for g in o.param_groups:
                    g["lr"] = base * min(1.0, (step + 1) / 300) * 0.5 * (1 + math.cos(math.pi * frac))
            s1 = tp[bi, 0]
            rr = tp[bi, 1]
            keys = s1
            if hn is not None:
                h = hn[bi]
                h = h[h >= 0]
                keys = np.concatenate([s1, h])
            rows = np.concatenate([rr, keys])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                z, zn, za, ea = model(*store.batch(rows))
            B = len(bi)
            zr, zk = z[:B], z[B:]
            T = model.log_temp.exp().clamp(0.01, 0.2)
            kid = torch.as_tensor(keys, device=z.device)
            sid = kid[:B]
            loss = _masked_ce(zr @ zk.T / T, kid, sid) + _masked_ce(zk[:B] @ zr.T / T, sid, sid)
            loss_n = _masked_ce(zn[:B] @ zn[B:2 * B].T / T, sid, sid)
            ok = (ea[:B] == 0) & (ea[B:2 * B] == 0)
            la = za[:B][ok] @ za[B:2 * B][ok].T / T
            loss_a = _masked_ce(la, sid[ok], sid[ok]) if ok.sum() > 1 else loss_n * 0
            total_loss = loss + 0.5 * (loss_n + loss_a)
            opt_s.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.step(opt_s)
            scaler.step(opt_d)
            scaler.update()
            step += 1
            if step % log_every == 0:
                print(f"[encoder] ep {ep} step {step}/{total} loss {loss.item():.4f} name {loss_n.item():.3f} "
                      f"addr {loss_a.item():.3f} T {T.item():.4f} {time.time() - t0:.0f}s", flush=True)
            if max_steps and step >= max_steps:
                break
        res = evaluate(model, store, device)
        res.update(epoch=ep, step=step, seconds=round(time.time() - t0))
        log.append(res)
        print(f"[encoder] eval after epoch {ep}: {json.dumps(res)}", flush=True)
        E.save(model, out / "encoder.pt")
        (out / "encoder_log.json").write_text(json.dumps(log, indent=1))
        torch.save({"model": model.state_dict(), "opt_s": opt_s.state_dict(), "opt_d": opt_d.state_dict(),
                    "scaler": scaler.state_dict(), "rng": rng.bit_generator.state, "step": step, "epoch": ep,
                    "log": log}, ck_path)
        if max_steps and step >= max_steps:
            break
    return model, log
