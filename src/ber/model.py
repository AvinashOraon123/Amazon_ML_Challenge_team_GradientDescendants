"""LightGBM matcher trained on an entity-disjoint subsample.

Leakage control: roles (train / valid-A / valid-B) are assigned per SOURCE-1 ENTITY
(`pipeline.entity_roles`), so all candidate pairs of an entity share one role and no
entity contributes to both fitting and validation. Each S2/S3 record has at most one true
S1, so its positive pair also sits in a single role.

  role 1  train    -> fit the trees
  role 2  valid-A  -> early stopping + threshold tuning
  role 3  valid-B  -> untouched final report (unbiased estimate of the leaderboard metric)

Memory: features are converted once to a float32 matrix, handed to LightGBM (which bins
them into ~1 byte per value) and freed, so a ~8M x 55 training set fits in ~2 GB.
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import feature_columns


def train_holdout(df: pd.DataFrame, params: dict, num_boost_round: int, early_stopping: int,
                  n_jobs: int, log_every: int = 100) -> lgb.Booster:
    feats = feature_columns(df)
    tr = (df["role"] == 1).to_numpy()
    va = (df["role"] == 2).to_numpy()
    t = time.time()
    Xtr = df.loc[tr, feats].to_numpy(dtype=np.float32)
    dtr = lgb.Dataset(Xtr, df.loc[tr, "label"].to_numpy(), feature_name=feats, free_raw_data=True).construct()
    del Xtr
    Xva = df.loc[va, feats].to_numpy(dtype=np.float32)
    dva = lgb.Dataset(Xva, df.loc[va, "label"].to_numpy(), reference=dtr, free_raw_data=True).construct()
    del Xva
    print(f"  datasets built: train={tr.sum():,} valid-A={va.sum():,} features={len(feats)} ({time.time() - t:.0f}s)")
    t = time.time()
    m = lgb.train({**params, "num_threads": n_jobs}, dtr, num_boost_round=num_boost_round,
                  valid_sets=[dva], valid_names=["validA"],
                  callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(log_every)])
    print(f"  trained: best_iter={m.best_iteration} validA_logloss={m.best_score['validA']['binary_logloss']:.5f} "
          f"({time.time() - t:.0f}s)")
    return m


def train_matrices(mats: dict, params: dict, num_boost_round: int, early_stopping: int, n_jobs: int,
                   log_every: int = 100) -> lgb.Booster:
    """Same as train_holdout, from pipeline.load_training_matrices (lower peak memory).
    The train matrix is released as soon as LightGBM has binned it."""
    feats = mats["features"]
    t = time.time()
    dtr = lgb.Dataset(mats[1]["X"], mats[1]["y"], weight=mats[1].get("w"), feature_name=feats,
                      free_raw_data=True).construct()
    mats[1]["X"] = None  # binned copy lives inside LightGBM now
    dva = lgb.Dataset(mats[2]["X"], mats[2]["y"], reference=dtr, free_raw_data=False).construct()
    print(f"  datasets built: train={dtr.num_data():,} valid-A={dva.num_data():,} features={len(feats)} ({time.time() - t:.0f}s)")
    t = time.time()
    m = lgb.train({**params, "num_threads": n_jobs}, dtr, num_boost_round=num_boost_round,
                  valid_sets=[dva], valid_names=["validA"],
                  callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(log_every)])
    print(f"  trained: best_iter={m.best_iteration} validA_logloss={m.best_score['validA']['binary_logloss']:.5f} "
          f"({time.time() - t:.0f}s)")
    return m


def predict_matrix(model: lgb.Booster, X: np.ndarray, n_jobs: int, chunk: int = 1_000_000) -> np.ndarray:
    out = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), chunk):
        out[s : s + chunk] = model.predict(X[s : s + chunk], num_iteration=model.best_iteration or None,
                                           num_threads=n_jobs)
    return out


def predict(model: lgb.Booster, df: pd.DataFrame, n_jobs: int, chunk: int = 1_000_000) -> np.ndarray:
    feats = model.feature_name()
    out = np.empty(len(df), dtype=np.float32)
    for s in range(0, len(df), chunk):
        X = df[feats].iloc[s : s + chunk].to_numpy(dtype=np.float32)
        out[s : s + chunk] = model.predict(X, num_iteration=model.best_iteration or None, num_threads=n_jobs)
    return out


def feature_importance(model: lgb.Booster) -> pd.DataFrame:
    return (pd.DataFrame({"feature": model.feature_name(),
                          "gain": model.feature_importance("gain"),
                          "split": model.feature_importance("split")})
            .sort_values("gain", ascending=False).reset_index(drop=True))


def save_meta(path, **meta) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=float)


def load_meta(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
