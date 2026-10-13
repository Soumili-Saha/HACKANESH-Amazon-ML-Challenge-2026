import os
import time

import numpy as np
import pandas as pd
import torch

MODEL = "intfloat/multilingual-e5-small"
PAIR_SHIFT = 1 << 26


def record_text(S):
    t = S.toks.fillna("").astype(str).values
    w = S.words.fillna("").astype(str).values
    n = S.nums.fillna("").astype(str).values
    return [f"query: {a} | {b} {c}".strip() for a, b, c in zip(t, w, n)]


@torch.no_grad()
def embed(texts, model_name=MODEL, batch=1024, max_len=64, device="cuda", out=None, out_rows=None):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()
    lens = np.fromiter((len(t) for t in texts), np.int32, len(texts))
    order = np.argsort(lens, kind="stable")
    dim = model.config.hidden_size
    if out is None:
        out = np.zeros((len(texts), dim), np.float16)
        out_rows = np.arange(len(texts))
    t0 = time.time()
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        v = torch.nn.functional.normalize(v.float(), dim=-1)
        out[out_rows[idx]] = v.cpu().numpy().astype(np.float16)
        if (s // batch) % 500 == 0:
            print(f"[embed] {s + len(idx):,}/{len(texts):,} ({time.time() - t0:.0f}s)", flush=True)
    return out


def embeddings(paths, split, S, rows=None, **kw):
    fp = paths.out(split, "emb.npy")
    if os.path.exists(fp):
        return np.load(fp, mmap_mode="r")
    from transformers import AutoConfig
    dim = AutoConfig.from_pretrained(kw.get("model_name", MODEL)).hidden_size
    rows = np.arange(len(S)) if rows is None else np.asarray(rows)
    texts = record_text(S.iloc[rows] if len(rows) < len(S) else S)
    tmp = fp + ".part.npy"
    emb = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(S), dim))
    embed(texts, out=emb, out_rows=rows, **kw)
    emb.flush()
    del emb
    os.replace(tmp, fp)
    return np.load(fp, mmap_mode="r")


def pair_cosine(emb, i1, i2, chunk=2_000_000):
    out = np.empty(len(i1), np.float32)
    for s in range(0, len(i1), chunk):
        a = emb[i1[s:s + chunk]].astype(np.float32)
        b = emb[i2[s:s + chunk]].astype(np.float32)
        out[s:s + chunk] = (a * b).sum(1)
    return out


@torch.no_grad()
def knn_pairs(emb, S, k=20, q_chunk=1024, p_chunk=1_000_000, query_rows=None, pool_rows=None, device="cuda"):
    """Top-k S2/S3 neighbours of every S1 within its country, by cosine."""
    cty = S.country.astype(str).values
    is1 = (S.src == "S1").values
    qmask = is1 if query_rows is None else np.isin(np.arange(len(S)), query_rows) & is1
    pmask = ~is1 if pool_rows is None else np.isin(np.arange(len(S)), pool_rows) & ~is1
    res_i1, res_i2, res_s, res_r = [], [], [], []
    torch.cuda.empty_cache()
    for c in np.unique(cty[qmask]):
        q = np.flatnonzero(qmask & (cty == c))
        pool = np.flatnonzero(pmask & (cty == c))
        if len(q) == 0 or len(pool) == 0:
            continue
        kk = min(k, len(pool))
        t0 = time.time()
        P = [torch.from_numpy(np.ascontiguousarray(emb[pool[s:s + p_chunk]])).to(device) for s in range(0, len(pool), p_chunk)]
        offs = np.r_[0, np.cumsum([len(x) for x in P])]
        for s in range(0, len(q), q_chunk):
            qi = q[s:s + q_chunk]
            Q = torch.from_numpy(np.ascontiguousarray(emb[qi])).to(device)
            best_s = torch.full((len(qi), kk), -2.0, device=device)
            best_i = torch.zeros((len(qi), kk), dtype=torch.long, device=device)
            for pi, Pc in enumerate(P):
                v, ix = (Q @ Pc.T).topk(min(kk, Pc.shape[0]), dim=1)
                v = v.float()
                cat_s = torch.cat([best_s, v], 1)
                cat_i = torch.cat([best_i, ix + int(offs[pi])], 1)
                best_s, sel = cat_s.topk(kk, dim=1)
                best_i = cat_i.gather(1, sel)
            bi = best_i.cpu().numpy()
            res_i1.append(np.repeat(qi, kk))
            res_i2.append(pool[bi].ravel())
            res_s.append(best_s.cpu().numpy().ravel().astype(np.float32))
            res_r.append(np.tile(np.arange(kk, dtype=np.int16), len(qi)))
        del P
        torch.cuda.empty_cache()
        print(f"[knn] {c}: {len(q):,} S1 x {len(pool):,} pool ({time.time() - t0:.0f}s)", flush=True)
    if not res_i1:
        return pd.DataFrame({"i1": [], "i2": [], "cos": [], "drank": []})
    return pd.DataFrame({"i1": np.concatenate(res_i1).astype(np.int64), "i2": np.concatenate(res_i2).astype(np.int64),
                         "cos": np.concatenate(res_s), "drank": np.concatenate(res_r)})


def new_pairs(knn, CE, k_new=10):
    have = np.sort(CE.i1.values.astype(np.int64) * PAIR_SHIFT + CE.i2.values.astype(np.int64))
    code = knn.i1.values * PAIR_SHIFT + knn.i2.values
    pos = np.clip(np.searchsorted(have, code), 0, max(len(have) - 1, 0))
    fresh = ~((len(have) > 0) & (have[pos] == code))
    D = knn[fresh & (knn.drank.values < k_new)].reset_index(drop=True)
    return D
