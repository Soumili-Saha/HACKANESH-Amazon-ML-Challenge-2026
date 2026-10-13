import json

import numpy as np

from .candidates import ORIGIN_DENSE
from .stack import logit

GAMMAS = (0.8, 1.0, 1.25, 1.5)
LAMS = (-0.1, -0.05, 0.0, 0.05, 0.1, 0.2)
MISSES = (0.0, 0.1, 0.25)
RESCUE = (None, 0.2, 0.3, 0.4, 0.5)


def hold_truth(er, paths, S, gt, fold):
    import os
    G = er.gt_arrays(S, gt)
    in_hold = fold == paths.hold
    sub = paths.out("train", "subset_s1.npy")
    if os.path.exists(sub):
        m = np.zeros(len(S), bool)
        m[np.load(sub)] = True
        in_hold = in_hold & m
    keep_t = in_hold[G["ti1"]]
    return dict(G, s1_rows=G["s1_rows"][in_hold[G["s1_rows"]]], ti1=G["ti1"][keep_t], ti2=G["ti2"][keep_t])


def rescue(er, K, p, keep, thr):
    if thr is None:
        return keep
    i1, i2 = K.i1.values.astype(np.int64), K.i2.values.astype(np.int64)
    used = np.zeros(int(i2.max()) + 1 if len(i2) else 1, bool)
    used[i2[keep]] = True
    best = er.assign_best(i1, i2, p)
    add = ~keep & best & (p >= thr) & ~used[i2]
    return keep | add


def tune(er, paths, K, S, gt, fold, p_stack):
    h = np.flatnonzero(K.fold.values == paths.hold)
    Kh = K.iloc[h].reset_index(drop=True)
    ph, p0 = p_stack[h], K.p.values[h]
    Gh = hold_truth(er, paths, S, gt, fold)
    base_rows = Kh.origin.values != ORIGIN_DENSE
    Kb = Kh[base_rows].reset_index(drop=True)
    base = max(er.score_keep(Kb, er.apply_decision(Kb, p0[base_rows], {"method": "expected", "gamma": g, "lam": l, "miss": m}), Gh)["F05"]
               for g in (1.0,) for l in LAMS for m in MISSES)
    grid = []
    for g in GAMMAS:
        for m in MISSES:
            for l in LAMS:
                d = {"method": "expected", "gamma": g, "lam": l, "miss": m}
                k = er.apply_decision(Kh, ph, d)
                for t in RESCUE:
                    kr = rescue(er, Kh, ph, k, t)
                    grid.append({**d, "rescue": t, "F05": er.score_keep(Kh, kr, Gh)["F05"]})
    grid.sort(key=lambda r: -r["F05"])
    best = grid[0]
    keep = rescue(er, Kh, ph, er.apply_decision(Kh, ph, {k: best[k] for k in ("method", "gamma", "lam", "miss")}), best["rescue"])
    rep = er.score_keep(Kh, keep, Gh)
    out = {"decision": {k: best[k] for k in ("method", "gamma", "lam", "miss")}, "rescue": best["rescue"],
           "F05_hold": rep["F05"], "F05_hold_by_country": rep["F05_by_country"], "F05_hold_baseline_lgbm": base,
           "top10": grid[:10]}
    json.dump(out, open(paths.out("train", "decision.json"), "w"), indent=1)
    return out


def shift_unseen(K, S, p, unseen_countries, delta):
    if not delta or not len(unseen_countries):
        return p
    cty = S.country.astype(str).values[K.i1.values]
    m = np.isin(cty, list(unseen_countries))
    q = p.astype(np.float64).copy()
    q[m] = 1 / (1 + np.exp(-(logit(q[m]) + delta)))
    return q.astype(np.float32)


def baseline_keep(K, S, path):
    """Rows of K that the pipeline's own submission file matched."""
    import pandas as pd
    m = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    m = m.assign(pool_id=m.matched_entity_ids.str.split(",")).explode("pool_id")
    m = m[m.pool_id.fillna("") != ""]
    idx = pd.Index(S.entity_id.astype(str).values)
    a, b = idx.get_indexer(m.source1_entity_id.values), idx.get_indexer(m.pool_id.values)
    ok = (a >= 0) & (b >= 0)
    want = np.unique(a[ok].astype(np.int64) * (1 << 26) + b[ok].astype(np.int64))
    code = K.i1.values.astype(np.int64) * (1 << 26) + K.i2.values.astype(np.int64)
    keep = np.isin(code, want)
    return keep, int(len(want)), int(keep.sum())


def finalize(er, paths, K, S, p_stack, unseen_countries=(), delta=-1.0, out_dir=None, baseline_path=None):
    dec = json.load(open(paths.out("train", "decision.json")))
    p = shift_unseen(K, S, p_stack, unseen_countries, delta)
    keep = rescue(er, K, p, er.apply_decision(K, p, dec["decision"]), dec["rescue"])
    if baseline_path and len(unseen_countries):
        bk, n_want, n_found = baseline_keep(K, S, baseline_path)
        un = np.isin(S.country.astype(str).values[K.i1.values], list(unseen_countries))
        keep = np.where(un, bk, keep)
        print(f"[decide] unseen-country rows taken from {baseline_path}: {int(bk[un].sum()):,} pairs "
              f"(baseline file pairs found in candidates: {n_found:,} of {n_want:,})", flush=True)
    out_dir = out_dir or paths.out("test", "submission")
    return er.write_submission(S, K, keep, out_dir), keep


def report(er, paths, K, S, gt, fold, sources, p_stack):
    h = np.flatnonzero(K.fold.values == paths.hold)
    Kh = K.iloc[h].reset_index(drop=True)
    Gh = hold_truth(er, paths, S, gt, fold)
    base = Kh.origin.values != ORIGIN_DENSE

    def best_f(C, p):
        return max(er.score_keep(C, er.apply_decision(C, p, {"method": "expected", "gamma": 1.0, "lam": l, "miss": m}), Gh)["F05"]
                   for l in LAMS for m in MISSES)

    rows = []
    Kb = Kh[base].reset_index(drop=True)
    rows.append(("pipeline LightGBM, pipeline candidates", best_f(Kb, K.p.values[h][base])))
    rows.append(("pipeline LightGBM, + dense candidates", best_f(Kh, K.p.values[h])))
    for name, v in sorted(sources.items()):
        q = v[h].copy()
        miss = np.isnan(q)
        q[miss] = K.p.values[h][miss]
        rows.append((f"{name} (LightGBM where not scored)", best_f(Kh, q)))
    rows.append(("stacker", best_f(Kh, p_stack[h])))
    dec = json.load(open(paths.out("train", "decision.json")))
    rows.append(("stacker + tuned decision + rescue", dec["F05_hold"]))
    out = {"rows": [{"model": a, "F05_hold": float(b)} for a, b in rows],
           "baseline": float(rows[0][1]), "final": float(dec["F05_hold"]),
           "final_by_country": dec["F05_hold_by_country"], "n_hold_s1": int(len(Gh["s1_rows"]))}
    json.dump(out, open(paths.out("train", "validation.json"), "w"), indent=1)
    return out
