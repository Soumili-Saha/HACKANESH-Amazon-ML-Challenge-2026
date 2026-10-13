import json
import os

import numpy as np
import pandas as pd

from .context import CAND_COLS

ORIGIN_BLOCK, ORIGIN_EXPAND, ORIGIN_DENSE = 0, 1, 2
DENSE_BIT = 1 << 6


def dense_frame(er, D, S):
    is_pool = S.part.astype(str).str.endswith("|POOL").to_numpy(dtype=bool)
    F = pd.DataFrame({"i1": D.i1.values.astype(np.int64), "i2": D.i2.values.astype(np.int64),
                      "bits": np.full(len(D), DENSE_BIT, np.int16), "knn": np.zeros(len(D), bool)})
    if len(F):
        F["sn"], F["sa"] = er._pair_sims(F, S)
    else:
        F["sn"], F["sa"] = np.zeros(0, np.float32), np.zeros(0, np.float32)
    F["pool"] = is_pool[F.i2.values]
    F["exp_sib"] = np.zeros(len(F), np.int8)
    F["exp_kaddr"] = np.zeros(len(F), np.int8)
    return F


def stage1_scores(paths, ctx, X):
    import lightgbm as lgb
    if len(X) == 0:
        return np.zeros(0, np.float32)
    if ctx.split == "test":
        return ctx.extra["m1"].predict(X).astype(np.float32)
    iters = json.load(open(f"{paths.v4_dir}/iters_s1.json"))["s1_iters"]
    f = ctx.extra["dense_fold"]
    out = np.zeros(len(X), np.float32)
    for k, it in enumerate(iters):
        m = ctx.extra.get(f"s1_fold{k}") or lgb.Booster(model_file=f"{paths.v4_dir}/s1_fold{k}.txt")
        sel = f == k
        if sel.any():
            out[sel] = m.predict(X[sel], num_iteration=it)
    return out


def row_features(er, cand, S, res, rows, n_workers):
    er._setup_features(cand, S, res)
    try:
        return er.features_for_rows(np.asarray(rows, np.int64), n_workers=n_workers)
    finally:
        er._FX.clear()


def build(paths, er, ctx, D, emb, s1_keep=None, gbm_min_p=0.001, n_workers=4):
    from .dense import pair_cosine
    S = ctx.S
    CE = ctx.CE
    if s1_keep is None:
        ce_idx = np.arange(len(CE))
        CEu = CE
    else:
        ce_idx = np.flatnonzero(np.isin(CE.i1.values, s1_keep))
        CEu = CE.iloc[ce_idx].reset_index(drop=True)
    base = CEu[CAND_COLS].copy()
    base["origin"] = np.where(ce_idx < ctx.n_blk, ORIGIN_BLOCK, ORIGIN_EXPAND).astype(np.int8)
    base["ce_row"] = ce_idx
    base["p"] = ctx.p[ce_idx]
    base["cos"] = pair_cosine(emb, CEu.i1.values.astype(np.int64), CEu.i2.values.astype(np.int64))
    base["drank"] = np.int16(-1)

    Dc = dense_frame(er, D, S)
    if ctx.split == "train" and len(Dc):
        keep = er.realistic_mask(Dc, S, ctx.gt)
        Dc, D = Dc[keep].reset_index(drop=True), D[keep].reset_index(drop=True)
    if s1_keep is not None and len(Dc):
        ks = np.isin(Dc.i1.values, s1_keep)
        Dc, D = Dc[ks].reset_index(drop=True), D[ks].reset_index(drop=True)

    n_feat = len(er.FEATURES_S1)
    XD = er.features_for_new(CEu, Dc, S, ctx.res, n_workers=n_workers) if len(Dc) else np.zeros((0, n_feat), np.float32)
    if ctx.split == "train":
        ctx.extra["dense_fold"] = ctx.fold[Dc.i1.values]
    pD = stage1_scores(paths, ctx, XD)
    dense = Dc[CAND_COLS].copy()
    dense["origin"] = np.int8(ORIGIN_DENSE)
    dense["ce_row"] = -1
    dense["p"] = pD
    dense["cos"] = D.cos.values.astype(np.float32)
    dense["drank"] = D.drank.values.astype(np.int16)

    K = pd.concat([base, dense], ignore_index=True)
    del base, dense
    if ctx.split == "train":
        yD = er.label_pairs(Dc, S, ctx.gt).astype(np.int8) if len(Dc) else np.zeros(0, np.int8)
        K["y"] = np.r_[ctx.y[ce_idx], yD]
        K["fold"] = ctx.fold[K.i1.values]

    n_base = len(CEu)
    rows = np.flatnonzero(K.p.values >= gbm_min_p)
    from_ce = rows[rows < n_base]
    from_d = rows[rows >= n_base] - n_base
    XC = row_features(er, CEu, S, ctx.res, from_ce, n_workers) if len(from_ce) else np.zeros((0, n_feat), np.float32)
    X = np.vstack([np.asarray(XC, np.float32), np.asarray(XD, np.float32)[from_d]])
    del XC

    K.to_parquet(paths.out(ctx.split, "cands.parquet"), index=False)
    np.save(paths.out(ctx.split, "gbm_rows.npy"), rows)
    np.save(paths.out(ctx.split, "gbm_X.npy"), X)
    return K, rows, X


def load(paths, split):
    K = pd.read_parquet(paths.out(split, "cands.parquet"))
    rows = np.load(paths.out(split, "gbm_rows.npy"))
    X = np.load(paths.out(split, "gbm_X.npy"), mmap_mode="r")
    return K, rows, X
