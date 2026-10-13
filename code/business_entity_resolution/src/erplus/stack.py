import glob
import json
import os

import numpy as np
import pandas as pd

from .candidates import ORIGIN_DENSE, ORIGIN_EXPAND

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)
ROUNDS = 400
PMIN = 0.001


def logit(x):
    x = np.clip(np.asarray(x, np.float64), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def load_sources(paths, split, K, S=None, unseen_countries=()):
    src = {}
    for fp in sorted(glob.glob(paths.out(split, "score_*.npy"))):
        name = os.path.basename(fp)[6:-4]
        if name.endswith("_st"):
            continue
        src[name] = np.load(fp).astype(np.float32)
    if split == "test" and len(unseen_countries) and S is not None:
        cty = S.country.astype(str).values[K.i1.values]
        un = np.isin(cty, list(unseen_countries))
        for name in list(src):
            st = paths.out(split, f"score_{name}_st.npy")
            if os.path.exists(st):
                v = np.load(st).astype(np.float32)
                m = un & ~np.isnan(v)
                src[name][m] = v[m]
    return src


def _group_stats(i1, v, name, F):
    d = pd.DataFrame({"i1": i1, "v": v})
    g = d.groupby("i1", sort=False).v
    mx = g.transform("max").values
    r = g.rank(ascending=False, method="first").values
    F[f"{name}_rank"] = r
    F[f"{name}_gap"] = mx - v
    F[f"{name}_max"] = mx
    F[f"{name}_npos"] = (d.v > 0).groupby(d.i1, sort=False).transform("sum").values
    F[f"{name}_second"] = d.v.where(r == 2).groupby(d.i1, sort=False).transform("max").fillna(-14).values


def features(K, rows, sources, S):
    i1, i2 = K.i1.values[rows], K.i2.values[rows]
    F = {"lp": logit(K.p.values[rows]), "cos": K.cos.values[rows].astype(np.float64),
         "drank": K.drank.values[rows].astype(np.float64),
         "is_dense": (K.origin.values[rows] == ORIGIN_DENSE).astype(np.float64),
         "is_expand": (K.origin.values[rows] == ORIGIN_EXPAND).astype(np.float64)}
    ce_names = [n for n in sources if n != "xgb"]
    for n, v in sources.items():
        F[f"l_{n}"] = logit(v[rows])
        F[f"l_{n}"][np.isnan(v[rows])] = np.nan
    if ce_names:
        L = np.vstack([F[f"l_{n}"] for n in ce_names])
        with np.errstate(all="ignore"):
            F["l_ce_mean"] = np.nanmean(L, 0)
        F["n_ce"] = (~np.isnan(L)).sum(0).astype(np.float64)
    parts = [F["lp"]] + ([F["l_xgb"]] if "l_xgb" in F else []) + ([F["l_ce_mean"]] if "l_ce_mean" in F else [])
    with np.errstate(all="ignore"):
        z = np.nanmean(np.vstack(parts), 0)
    F["l_comb"] = z
    F["n_cand"] = pd.Series(i1).groupby(i1).transform("size").values.astype(np.float64)
    for n in ["lp", "cos", "l_comb"] + (["l_xgb"] if "l_xgb" in F else []) + (["l_ce_mean"] if "l_ce_mean" in F else []):
        v = np.nan_to_num(F[n], nan=-14.0)
        _group_stats(i1, v, n, F)
    zc = np.nan_to_num(z, nan=-14.0)
    a = pd.DataFrame({"i2": i2, "z": zc}).sort_values(["i2", "z"], ascending=[True, False])
    rk = a.groupby("i2", sort=False).cumcount().values
    best = a[rk == 0].set_index("i2").z
    second = a[rk == 1].set_index("i2").z
    nn = a.groupby("i2", sort=False).size()
    s2 = pd.Series(i2)
    b1 = s2.map(best).fillna(-14).values
    b2 = s2.map(second).fillna(-14).values
    other = np.where(zc >= b1, b2, b1)
    F["pool_best_other"] = other
    F["pool_margin"] = zc - other
    F["pool_is_top"] = (zc >= b1).astype(np.float64)
    F["pool_n"] = s2.map(nn).fillna(1).values.astype(np.float64)
    F["b_s3"] = (S.src == "S3").values[i2].astype(np.float64)
    F["b_has_addr"] = S.has_addr.values.astype(bool)[i2].astype(np.float64) if "has_addr" in S.columns else np.nan
    cols = list(F)
    return np.column_stack([np.asarray(F[c], np.float32) for c in cols]), cols


def stack_rows(paths, K, split):
    r = np.flatnonzero(K.p.values >= PMIN)
    if split == "train":
        r = r[K.fold.values[r] == paths.hold]
    return r


def fit(paths, K, S, sources, inner=4):
    import lightgbm as lgb
    rows = stack_rows(paths, K, "train")
    X, cols = features(K, rows, sources, S)
    y = K.y.values[rows]
    grp = (K.i1.values[rows].astype(np.int64) * 2654435761 % 2 ** 32) % inner
    oof = np.zeros(len(rows), np.float32)
    for f in range(inner):
        tr, va = grp != f, grp == f
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=cols), num_boost_round=ROUNDS)
        oof[va] = m.predict(X[va])
    m = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=cols), num_boost_round=ROUNDS)
    m.save_model(paths.out("train", "stacker.txt"))
    json.dump({"cols": cols, "sources": sorted(sources)}, open(paths.out("train", "stacker.json"), "w"), indent=1)
    p = K.p.values.astype(np.float32).copy()
    p[rows] = oof
    np.save(paths.out("train", "p_stack.npy"), p)
    return p, rows


def apply(paths, K, S, sources):
    import lightgbm as lgb
    meta = json.load(open(paths.out("train", "stacker.json")))
    missing = [s for s in meta["sources"] if s not in sources]
    for s in missing:
        sources[s] = np.full(len(K), np.nan, np.float32)
    sources = {k: sources[k] for k in meta["sources"]}
    rows = stack_rows(paths, K, "test")
    X, cols = features(K, rows, sources, S)
    assert cols == meta["cols"], "stacker feature set differs between train and test"
    m = lgb.Booster(model_file=paths.out("train", "stacker.txt"))
    p = K.p.values.astype(np.float32).copy()
    p[rows] = m.predict(X)
    np.save(paths.out("test", "p_stack.npy"), p)
    return p, rows
