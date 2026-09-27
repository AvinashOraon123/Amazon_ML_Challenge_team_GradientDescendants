"""Stage 5: pair classifier = LightGBM + MLP ensemble, trained with 5-fold cross-validation.

Training rows are candidate pairs whose Source-1 entity is in the validation holdout: the encoder
never saw those entities, so their embedding features behave like the test set's. Folds are split
by record, and every candidate of a record lives in the same fold, so the one-entity-per-record step
and the decision rule can be tuned on genuinely out-of-fold probabilities.
"""
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import torch
import torch.nn as nn

from . import decide
from .features import feature_columns

NFOLD = 5
LGB_PARAMS = dict(objective="binary", learning_rate=0.06, num_leaves=127, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, verbose=-1, num_threads=0)


def _backend():
    """XGBoost on the GPU when available (millions of rows in minutes on a T4), otherwise LightGBM on CPU.
    Override with BER_GBDT=lgb|xgb."""
    want = os.environ.get("BER_GBDT")
    if want in ("lgb", "xgb"):
        return want
    try:
        import xgboost  # noqa: F401
        return "xgb" if torch.cuda.is_available() else "lgb"
    except ImportError:
        return "lgb"


BACKEND = _backend()
EXT = ".txt" if BACKEND == "lgb" else ".json"

# Stage-1 GBDT variants, averaged (BER_GB_VARIANTS=lg127,d10,...). Each is a tree shape + learning rate.
VARIANTS = {
    "lg127": dict(leaves=127, lr=0.06, depth=0),     # leaf-wise, 127 leaves (the default)
    "lg255": dict(leaves=255, lr=0.04, depth=0),     # bigger leaf-wise trees, slower rate
    "d10": dict(leaves=0, lr=0.05, depth=10),        # depth-wise trees, depth 10
}
GB_VARIANTS = [v for v in os.environ.get("BER_GB_VARIANTS", "lg127").split(",") if v]
# The MLP never won the blend (weight 0 in every full run) and holds GPU memory the GBDT needs: BER_MLP=0 skips it.
USE_MLP = os.environ.get("BER_MLP", "1") != "0"


def free_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clean(X):
    """Finite float32 matrix (XGBoost rejects inf; NaN/inf also hurt the MLP)."""
    return np.nan_to_num(np.asarray(X, dtype=np.float32), nan=-9.0, posinf=9.0, neginf=-9.0)


def gb_train(X, y, Xva=None, yva=None, leaves=127, lr=0.06, rounds=1500, early=50, depth=0):
    if BACKEND == "lgb":
        params = {**LGB_PARAMS, "num_leaves": leaves or 1023, "learning_rate": lr}
        if depth:
            params["max_depth"] = depth
        dtr = lgb.Dataset(X, y, free_raw_data=True)
        if Xva is None:
            return lgb.train(params, dtr, num_boost_round=rounds)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        return lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(early, verbose=False)])
    import xgboost as xgb
    params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                  device="cuda" if torch.cuda.is_available() else "cpu",
                  eta=lr, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, max_bin=256)
    if depth:
        params.update(max_depth=depth)
    else:
        params.update(max_depth=0, grow_policy="lossguide", max_leaves=leaves)
    dtr = xgb.QuantileDMatrix(X, y, max_bin=256)
    if Xva is None:
        return xgb.train(params, dtr, num_boost_round=rounds)
    dva = xgb.QuantileDMatrix(Xva, yva, ref=dtr)
    return xgb.train(params, dtr, num_boost_round=rounds, evals=[(dva, "va")], early_stopping_rounds=early,
                     verbose_eval=False)


def gb_iter(b):
    if BACKEND == "lgb":
        return b.best_iteration or b.current_iteration()
    return (getattr(b, "best_iteration", None) or b.num_boosted_rounds() - 1) + 1


def gb_predict(b, X, chunk=2_000_000):
    out = np.empty(len(X), np.float32)
    for a in range(0, len(X), chunk):
        if BACKEND == "lgb":
            out[a:a + chunk] = b.predict(X[a:a + chunk], num_iteration=b.best_iteration or None)
        else:
            out[a:a + chunk] = b.inplace_predict(X[a:a + chunk], iteration_range=(0, gb_iter(b)))
    return out


def gb_save(b, path):
    b.save_model(str(path))


def gb_load(path):
    if BACKEND == "lgb":
        return lgb.Booster(model_file=str(path))
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(str(path))
    b.set_param({"device": "cuda" if torch.cuda.is_available() else "cpu"})
    return b


class MLP(nn.Module):
    def __init__(self, d, h=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.GELU(),
            nn.Linear(h // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _scale_fit(X):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def train_mlp(Xtr, ytr, Xva, yva, device, epochs=6, bs=8192, lr=2e-3, log=print):
    mu, sd = _scale_fit(Xtr)
    tx = lambda X: torch.from_numpy(np.clip((X - mu) / sd, -10, 10).astype(np.float32))
    Xt, yt = tx(Xtr).to(device), torch.from_numpy(ytr.astype(np.float32)).to(device)
    model = MLP(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    steps = epochs * ((len(Xt) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        for a in range(0, len(Xt), bs):
            i = perm[a:a + bs]
            loss = lossf(model(Xt[i]), yt[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    state = {"model": model.state_dict(), "mu": mu, "sd": sd, "d": Xtr.shape[1]}
    return state, predict_mlp(state, Xva, device)


@torch.no_grad()
def predict_mlp(state, X, device, bs=65536):
    model = MLP(state["d"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    out = []
    for a in range(0, len(X), bs):
        x = torch.from_numpy(np.clip((X[a:a + bs] - state["mu"]) / state["sd"], -10, 10).astype(np.float32)).to(device)
        out.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def train_cv(feat: pl.DataFrame, out_dir, device, log=print):
    """feat: features + label + fold + is_hold. Saves fold models, returns OOF probabilities per model."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = feature_columns(feat.drop("is_hold"))
    (out / "feature_columns.json").write_text(json.dumps(cols))
    X = clean(feat.select(cols).to_numpy().astype(np.float32))
    y = feat["label"].to_numpy().astype(np.float32)
    fold = feat["fold"].to_numpy()
    hold = feat["is_hold"].to_numpy()
    oof = {"lgb": np.zeros(len(X), np.float32), "mlp": np.zeros(len(X), np.float32)}
    iters = {}
    for k in range(NFOLD):
        t0 = time.time()
        tr, va = (fold != k) & hold, fold == k
        pv = np.zeros(int(va.sum()), np.float32)
        for vname in GB_VARIANTS:                     # average of the configured tree variants
            mpath = out / f"gb_fold{k}_{vname}{EXT}"
            if mpath.exists():
                booster = gb_load(mpath)
            else:
                booster = gb_train(X[tr], y[tr], X[va & hold], y[va & hold], **VARIANTS[vname])
                gb_save(booster, mpath)
            pv += gb_predict(booster, X[va]) / len(GB_VARIANTS)
            iters.setdefault(vname, []).append(gb_iter(booster))
        oof["lgb"][va] = pv
        mlp_path = out / f"mlp_fold{k}.pt"
        if not USE_MLP:
            pass
        elif mlp_path.exists():
            state = torch.load(mlp_path, weights_only=False)
            oof["mlp"][va] = predict_mlp(state, X[va], device)
        else:
            state, oof["mlp"][va] = train_mlp(X[tr], y[tr], X[va], y[va], device)
            torch.save(state, mlp_path)
            del state
        free_gpu()
        log(f"[matcher] fold {k}: {BACKEND} {GB_VARIANTS} trees {[v[-1] for v in iters.values()]} ({time.time() - t0:.0f}s)")
    for vname in GB_VARIANTS:     # one model per variant on all held-out rows: cheap to apply to the test set
        full = out / f"gb_full_{vname}{EXT}"
        if not full.exists():
            rounds = int(np.mean(iters[vname]) * 1.1)
            gb_save(gb_train(X[hold], y[hold], rounds=rounds, **VARIANTS[vname]), full)
            log(f"[matcher] full {BACKEND} {vname} model: {rounds} trees")
    return oof


def predict(X, model_dir, device, w_lgb=0.5):
    """Test-time probabilities: the full GBDT model (fold average as fallback) and the fold-averaged MLP.
    Models with zero blend weight are skipped."""
    X = clean(X)
    model_dir = Path(model_dir)
    p_lgb = np.zeros(len(X), np.float32)
    p_mlp = np.zeros(len(X), np.float32)
    if w_lgb > 0:
        for vname in GB_VARIANTS:
            full = model_dir / f"gb_full_{vname}{EXT}"
            if full.exists():
                p_lgb += gb_predict(gb_load(full), X) / len(GB_VARIANTS)
            else:
                for k in range(NFOLD):
                    p_lgb += gb_predict(gb_load(model_dir / f"gb_fold{k}_{vname}{EXT}"), X) / (NFOLD * len(GB_VARIANTS))
    if w_lgb < 1:
        for k in range(NFOLD):
            p_mlp += predict_mlp(torch.load(model_dir / f"mlp_fold{k}.pt", weights_only=False), X, device) / NFOLD
    return {"lgb": p_lgb, "mlp": p_mlp}


def blend(ps, w):
    return w * ps["lgb"] + (1 - w) * ps["mlp"]


def fit_and_tune(feat, true_pairs, s1_eval, out_dir, device, log=print):
    """CV-train both models, choose blend weight + decision rule on OOF F0.5 over holdout entities, then
    try the stage-2 stacker on top and keep it only if it improves OOF F0.5."""
    oof = train_cv(feat, out_dir, device, log)
    base = feat.select("s1_rid", "rid")
    report = {}
    best = None
    for w in ((0.0, 0.3, 0.5, 0.7, 1.0) if USE_MLP else (1.0,)):
        pred = base.with_columns(pl.Series("p", blend(oof, w)))
        rule, results = decide.tune(pred, true_pairs, s1_eval, log=lambda *_: None)
        score = max(s for _, s in results)
        report[f"w_lgb={w}"] = {"rule": rule, "f05": score}
        log(f"[matcher] blend w_lgb={w}: best OOF F0.5 {score:.5f} with {rule}")
        if best is None or score > best[0]:
            best = (score, w, rule)
    score, w, rule = best
    p1 = blend(oof, w)
    s2 = stack_features(feat, p1)
    p2 = train_stage2(s2, feat["label"].to_numpy().astype(np.float32), feat["fold"].to_numpy(),
                      feat["is_hold"].to_numpy(), out_dir, log)
    rule2, results2 = decide.tune(base.with_columns(pl.Series("p", p2)), true_pairs, s1_eval, log=lambda *_: None)
    score2 = max(s for _, s in results2)
    report["stage2"] = {"rule": rule2, "f05": score2}
    log(f"[matcher] stage-2 stacker: best OOF F0.5 {score2:.5f} with {rule2} (stage 1: {score:.5f})")
    use2 = score2 > score
    cfg = {"w_lgb": w, "stage2": use2, "rule": rule2 if use2 else rule, "oof_f05": max(score, score2),
           "stage1_f05": score, "report": report}
    (Path(out_dir) / "decision.json").write_text(json.dumps(cfg, indent=1))
    pred = base.with_columns(pl.Series("p", p2 if use2 else p1))
    return cfg, pred


def country_transfer(feat, country_of_s1, true_pairs, s1_eval_all, log=print):
    """Proxy for the unseen-country (France) risk: train LightGBM on one country, test on the other."""
    cols = feature_columns(feat.drop("is_hold"))
    c = country_of_s1[feat["s1_rid"].to_numpy()]
    hold = feat["is_hold"].to_numpy()
    X = clean(feat.select(cols).to_numpy().astype(np.float32))
    y = feat["label"].to_numpy().astype(np.float32)
    res = {}
    for src_c in np.unique(c):
        for dst_c in np.unique(c):
            if src_c == dst_c:
                continue
            tr = hold & (c == src_c)
            b = gb_train(X[tr], y[tr], rounds=300)
            te = c == dst_c
            pred = feat.filter(pl.Series(te)).select("s1_rid", "rid").with_columns(pl.Series("p", gb_predict(b, X[te])))
            ev = s1_eval_all[country_of_s1[s1_eval_all] == dst_c]
            rule, results = decide.tune(pred, true_pairs, ev, log=lambda *_: None)
            res[f"{src_c}->{dst_c}"] = max(s for _, s in results)
            log(f"[matcher] country transfer {src_c} -> {dst_c}: F0.5 {res[f'{src_c}->{dst_c}']:.5f}")
    return res


# ---------------------------------------------------------------- stage 2 (stacking on competition)
STAGE2_BASE = ("z_cos", "za_cos", "zn_cos", "z_vs2", "core_tset", "addr_tset", "num_first_eq", "num_conflict",
               "addr_empty_2", "s1_ncand", "rec_ncand")


def stack_features(feat: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    """Competition features from stage-1 probabilities: how a pair compares with the other candidates
    of the same record and of the same Source-1 entity."""
    d = feat.select("s1_rid", "rid", *[c for c in STAGE2_BASE if c in feat.columns]).with_columns(pl.Series("p", p))
    return d.with_columns(
        (pl.col("p").max().over("rid")).alias("rec_pmax"),
        (pl.col("p").rank("ordinal", descending=True).over("rid")).cast(pl.Float32).alias("rec_prank"),
        ((pl.col("p") > 0.5).sum().over("rid")).cast(pl.Float32).alias("rec_n05"),
        (pl.col("p").sum().over("rid") - pl.col("p")).alias("rec_psum_other"),
        (pl.col("p").max().over("s1_rid")).alias("s1_pmax"),
        (pl.col("p").sum().over("s1_rid")).alias("s1_psum"),
        (pl.col("p").rank("ordinal", descending=True).over("s1_rid")).cast(pl.Float32).alias("s1_prank"),
        ((pl.col("p") > 0.5).sum().over("s1_rid")).cast(pl.Float32).alias("s1_n05"),
    ).with_columns(
        (pl.col("p") - pl.col("rec_pmax")).alias("p_minus_recmax"),
        (pl.col("p") / pl.col("s1_pmax").clip(1e-6, None)).alias("p_over_s1max"),
        # best probability among the record's OTHER entities
        pl.when(pl.col("rec_prank") == 1)
        .then((pl.col("p").sort(descending=True).over("rid", mapping_strategy="join").list.get(1, null_on_oob=True)))
        .otherwise(pl.col("rec_pmax")).fill_null(0.0).alias("rec_p_other_best"),
    )


def stage2_columns(s2: pl.DataFrame):
    return [c for c in s2.columns if c not in ("s1_rid", "rid")]


def train_stage2(s2: pl.DataFrame, y, fold, hold, out_dir, log=print):
    out = Path(out_dir)
    cols = stage2_columns(s2)
    (out / "stage2_columns.json").write_text(json.dumps(cols))
    X = clean(s2.select(cols).to_numpy().astype(np.float32))
    oof = np.zeros(len(X), np.float32)
    iters = []
    for k in range(NFOLD):
        tr, va = (fold != k) & hold, fold == k
        mpath = out / f"s2_fold{k}{EXT}"
        if mpath.exists():
            b = gb_load(mpath)
        else:
            b = gb_train(X[tr], y[tr], X[va & hold], y[va & hold], leaves=63)
            gb_save(b, mpath)
        oof[va] = gb_predict(b, X[va])
        iters.append(gb_iter(b))
        log(f"[matcher] stage-2 fold {k}: {iters[-1]} trees")
    full = out / f"s2_full{EXT}"
    if not full.exists():
        rounds = int(np.mean(iters) * 1.1)
        gb_save(gb_train(X[hold], y[hold], leaves=63, rounds=rounds), full)
    return oof


def predict_stage2(s2: pl.DataFrame, model_dir):
    cols = json.loads((Path(model_dir) / "stage2_columns.json").read_text())
    X = clean(s2.select(cols).to_numpy().astype(np.float32))
    full = Path(model_dir) / f"s2_full{EXT}"
    if full.exists():
        return gb_predict(gb_load(full), X)
    p = np.zeros(len(X), np.float32)
    for k in range(NFOLD):
        p += gb_predict(gb_load(Path(model_dir) / f"s2_fold{k}{EXT}"), X) / NFOLD
    return p
