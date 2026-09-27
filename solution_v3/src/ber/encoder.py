"""Hashed character n-gram bi-encoder (fastText-style bag, learned IDF-like weights, MLP head).

Input per record: three fixed-width uint8 byte matrices (name, skeleton, address). N-grams are
hashed on the GPU, so no tokenisation happens on the CPU at all. Output: an L2-normalised joint
embedding `z` used for candidate search, plus name-only / address-only embeddings used later as
matcher features.
"""
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    buckets_log2: int = 22          # 4M hash buckets shared by all fields (bucket 0 = padding)
    dim: int = 128                  # n-gram embedding width
    out_dim: int = 128              # joint embedding width (what the kNN search uses)
    field_dim: int = 64             # name-only / address-only embedding width
    hidden: int = 512
    name_len: int = 66              # byte widths incl. the leading/trailing space
    skel_len: int = 42
    addr_len: int = 114
    name_ngrams: tuple = (2, 3, 4, 5)
    skel_ngrams: tuple = (2, 3, 4)
    addr_ngrams: tuple = (3, 4, 5)


FIELDS = ("name", "skel", "addr")
_MIX = 0x9E3779B97F4A7C15 - (1 << 64)   # 64-bit golden-ratio constant as signed int64


def hash_ngrams(b: torch.Tensor, ns, salt: int, nbits: int):
    """b: (B, L) uint8 on device. Returns (B, P) int64 bucket ids with 0 for invalid windows."""
    x = b.long()
    L = x.shape[1]
    out = []
    for n in ns:
        if n > L:
            continue
        h = torch.zeros_like(x[:, : L - n + 1])
        for k in range(n):
            h = h * 257 + x[:, k: L - n + 1 + k]
        h = (h + (salt * 131 + n) * 1_000_003) * _MIX          # wraps mod 2^64, deterministic
        h = (h ^ (h >> 29)) & ((1 << nbits) - 1)
        h = torch.where(h == 0, torch.ones_like(h), h)         # keep 0 reserved for padding
        valid = x[:, n - 1:] != 0                              # windows ending inside the string
        out.append(torch.where(valid, h, torch.zeros_like(h)))
    return torch.cat(out, dim=1)


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        nb = 1 << cfg.buckets_log2
        self.emb = nn.Embedding(nb, cfg.dim, padding_idx=0, sparse=True)
        self.wlog = nn.Embedding(nb, 1, padding_idx=0, sparse=True)   # learned n-gram importance
        nn.init.normal_(self.emb.weight, std=0.05)
        nn.init.zeros_(self.wlog.weight)
        with torch.no_grad():
            self.emb.weight[0].zero_()
        self.ngrams = {"name": cfg.name_ngrams, "skel": cfg.skel_ngrams, "addr": cfg.addr_ngrams}
        d3 = 3 * cfg.dim + 3
        self.head = nn.Sequential(
            nn.Linear(d3, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU(),
            nn.Linear(cfg.hidden, cfg.out_dim),
        )
        self.name_head = nn.Sequential(nn.Linear(2 * cfg.dim, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, cfg.field_dim))
        self.addr_head = nn.Sequential(nn.Linear(cfg.dim, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, cfg.field_dim))
        self.log_temp = nn.Parameter(torch.tensor(-3.0))       # temperature ~0.05, learned

    def sparse_parameters(self):
        return [self.emb.weight, self.wlog.weight]

    def dense_parameters(self):
        sp = {id(p) for p in self.sparse_parameters()}
        return [p for p in self.parameters() if id(p) not in sp]

    def pool(self, b, field):
        ids = hash_ngrams(b, self.ngrams[field], FIELDS.index(field), self.cfg.buckets_log2)
        mask = ids != 0
        # flatten to the real n-grams only (padding is ~2/3 of positions): keeps sparse grads small
        counts = mask.sum(1)
        flat = ids[mask]
        offsets = torch.cumsum(counts, 0) - counts
        w = F.softplus(self.wlog(flat).squeeze(-1) + 1.0)
        wsum = torch.zeros(len(ids), device=ids.device, dtype=w.dtype).index_add_(
            0, torch.repeat_interleave(torch.arange(len(ids), device=ids.device), counts), w)
        v = F.embedding_bag(flat, self.emb.weight, offsets, per_sample_weights=w, mode="sum", sparse=True)
        v = v / wsum.clamp_min(1e-6)[:, None]
        empty = (counts == 0).float()[:, None]
        return v, empty

    def forward(self, name, skel, addr):
        vn, en = self.pool(name, "name")
        vs, es = self.pool(skel, "skel")
        va, ea = self.pool(addr, "addr")
        h = torch.cat([vn, vs, va, en, es, ea], dim=1)
        z = F.normalize(self.head(h).float(), dim=-1)
        zn = F.normalize(self.name_head(torch.cat([vn, vs], 1)).float(), dim=-1)
        za = F.normalize(self.addr_head(va).float(), dim=-1)
        return z, zn, za, ea.squeeze(1)


def save(model: Encoder, path):
    torch.save({"cfg": asdict(model.cfg), "state": model.state_dict()}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    cfg = EncoderConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})
    m = Encoder(cfg).to(device)
    m.load_state_dict(ck["state"])
    return m.eval()


@torch.no_grad()
def encode(model: Encoder, name, skel, addr, device, batch=16384, parts=("z",)):
    """Encode byte matrices (numpy/torch uint8) -> dict of fp16 CPU tensors."""
    outs = {p: [] for p in parts}
    for a in range(0, len(name), batch):
        args = [torch.as_tensor(m[a:a + batch]).to(device, non_blocking=True) for m in (name, skel, addr)]
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=device.startswith("cuda")):
            z, zn, za, _ = model(*args)
        got = {"z": z, "zn": zn, "za": za}
        for p in parts:
            outs[p].append(got[p].half().cpu())
    return {p: torch.cat(v) for p, v in outs.items()}
