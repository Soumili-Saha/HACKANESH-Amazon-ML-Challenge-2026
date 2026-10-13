import json

import numpy as np

from .candidates import ORIGIN_DENSE

PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", max_depth=8,
              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0)


def design(K, rows, X):
    extra = np.column_stack([K.cos.values[rows], K.drank.values[rows].astype(np.float32),
                             (K.origin.values[rows] == ORIGIN_DENSE).astype(np.float32)])
    return np.hstack([np.asarray(X, np.float32), extra.astype(np.float32)])


def fit(paths, K, rows, X, rounds=2000, early=100, device="cuda", seed=42):
    import xgboost as xgb
    Z = design(K, rows, X)
    y = K.y.values[rows]
    fold = K.fold.values[rows]
    tr = np.flatnonzero(fold != paths.hold)
    rng = np.random.RandomState(seed)
    s1 = K.i1.values[rows][tr]
    us = np.unique(s1)
    es = np.isin(s1, us[rng.rand(len(us)) < 0.1])
    dtr = xgb.DMatrix(Z[tr[~es]], label=y[tr[~es]])
    des = xgb.DMatrix(Z[tr[es]], label=y[tr[es]])
    m = xgb.train({**PARAMS, "device": device, "seed": seed}, dtr, num_boost_round=rounds,
                  evals=[(des, "es")], early_stopping_rounds=early, verbose_eval=200)
    m.save_model(paths.out("train", "xgb.json"))
    json.dump({"best_iteration": int(m.best_iteration)}, open(paths.out("train", "xgb_info.json"), "w"))
    return m


def score(paths, split, K, rows, X, model=None):
    import xgboost as xgb
    if model is None:
        model = xgb.Booster()
        model.load_model(paths.out("train", "xgb.json"))
    best = json.load(open(paths.out("train", "xgb_info.json")))["best_iteration"]
    out = np.full(len(K), np.nan, np.float32)
    Z = design(K, rows, X)
    for s in range(0, len(rows), 1_000_000):
        out[rows[s:s + 1_000_000]] = model.predict(xgb.DMatrix(Z[s:s + 1_000_000]), iteration_range=(0, best + 1))
    np.save(paths.out(split, "score_xgb.npy"), out)
    return out
