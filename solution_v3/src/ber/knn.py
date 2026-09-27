"""Exact top-k inner-product search with chunked GPU matrix multiplies (no FAISS dependency).

torch.topk over millions of columns per row is the bottleneck, so selection is two-level and still
exact: take the max of every group of G consecutive scores (a cheap bandwidth-bound reduction), pick
the top-k groups per row, and run the exact top-k only inside those k*G scores. The k best scores of
a row always lie inside its k best groups.

Search is always done inside one country: no true match crosses countries in the training data.
"""
import numpy as np
import torch

GROUP = 32


@torch.no_grad()
def topk(queries: torch.Tensor, keys: torch.Tensor, k: int, device: str, q_chunk=None):
    """queries (Nq, d), keys (Nk, d): fp16/fp32 CPU tensors, rows L2-normalised.

    Returns (idx int64 (Nq, k) into keys, scores float32 (Nq, k)), sorted by descending score.
    """
    nk = len(keys)
    k = min(k, nk)
    if len(queries) == 0 or k == 0:
        return np.zeros((len(queries), k), np.int64), np.zeros((len(queries), k), np.float32)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pad = (-nk) % GROUP
    K = torch.cat([keys, keys.new_zeros((pad, keys.shape[1]))]).to(device=device, dtype=dtype)
    ng = K.shape[0] // GROUP
    kg = min(k, ng)
    if q_chunk is None:   # keep the (q_chunk x Nk) score block around 1.5 GB
        q_chunk = max(64, min(16384, int(7.5e8 // max(K.shape[0], 1))))
    out_i = np.empty((len(queries), k), np.int64)
    out_s = np.empty((len(queries), k), np.float32)
    for a in range(0, len(queries), q_chunk):
        Q = queries[a:a + q_chunk].to(device=device, dtype=dtype)
        s = Q @ K.T
        if pad:
            s[:, nk:] = float("-inf")
        gmax = s.view(len(Q), ng, GROUP).amax(2)                          # (q, ng)
        top_g = gmax.topk(kg, dim=1).indices                              # (q, kg)
        cols = (top_g[:, :, None] * GROUP + torch.arange(GROUP, device=s.device)).reshape(len(Q), -1)
        cand = s.gather(1, cols)                                          # (q, kg*GROUP)
        ss, j = cand.topk(k, dim=1)
        out_i[a:a + len(Q)] = cols.gather(1, j).cpu().numpy()
        out_s[a:a + len(Q)] = ss.float().cpu().numpy()
    return out_i, out_s


def topk_by_group(q_rows, q_group, k_rows, k_group, emb, k, device):
    """Search each query row against the key rows of the same group (country).

    q_rows / k_rows: int arrays of row ids into `emb`; *_group: group label per row.
    Returns (nbr_rows (len(q_rows), k) int64 with -1 padding, scores float32 with -inf padding).
    """
    emb = torch.as_tensor(np.asarray(emb)) if not isinstance(emb, torch.Tensor) else emb
    nbr = np.full((len(q_rows), k), -1, np.int64)
    sc = np.full((len(q_rows), k), -np.inf, np.float32)
    for g in np.unique(q_group):
        qi = np.nonzero(q_group == g)[0]
        kr = k_rows[k_group == g]
        if len(kr) == 0:
            continue
        idx, s = topk(emb[torch.as_tensor(q_rows[qi])], emb[torch.as_tensor(kr)], k, device)
        kk = idx.shape[1]
        nbr[qi, :kk] = kr[idx]
        sc[qi, :kk] = s
    return nbr, sc
