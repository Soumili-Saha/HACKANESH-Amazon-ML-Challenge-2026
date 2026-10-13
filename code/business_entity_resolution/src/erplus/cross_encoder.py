import csv
import math
import os
import time

import numpy as np
import pandas as pd
import torch

from .candidates import ORIGIN_DENSE

READ_KW = dict(sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False, na_filter=False, encoding="utf-8")


def raw_texts(paths, split, ids):
    ids = set(ids)
    out = {}
    for k in (1, 2, 3):
        pq_names = [f"source{k}.parquet", f"train_source{k}.parquet"] if split == "train" else [f"test_source{k}.parquet"]
        df = None
        for n in pq_names:
            fp = os.path.join(paths.parquet_dir, n)
            if os.path.exists(fp):
                df = pd.read_parquet(fp, columns=["entity_id", "business_name", "business_address"])
                break
        if df is None:
            df = pd.read_csv(os.path.join(paths.data_dir, split, f"{split}_source{k}.tsv"),
                             usecols=["entity_id", "business_name", "business_address"], **READ_KW)
        df = df[df.entity_id.astype(str).isin(ids)]
        out.update(zip(df.entity_id.astype(str), df.business_name.fillna("").astype(str) + " | " +
                       df.business_address.fillna("").astype(str)))
    return out


def score_rows(K, lo=0.01, hi=0.99, dense_min=0.005):
    p = K.p.values
    band = (p >= lo) & (p <= hi)
    dense = (K.origin.values == ORIGIN_DENSE) & (p >= dense_min)
    return np.flatnonzero(band | dense)


def train_rows(paths, K, n_max=1_500_000, seed=0):
    p, fold = K.p.values, K.fold.values
    rng = np.random.RandomState(seed)
    tr = fold != paths.hold
    band = (p >= 0.005) & (p <= 0.995)
    pick = tr & (band | ((p < 0.005) & (rng.rand(len(p)) < 0.01)) | ((p > 0.995) & (rng.rand(len(p)) < 0.10)))
    rows = np.flatnonzero(pick)
    if len(rows) > n_max:
        rows = np.sort(rng.choice(rows, n_max, replace=False))
    return rows


class PairModel:
    def __init__(self, name, max_len=96, device="cuda"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).to(device)
        self.max_len, self.device = max_len, device

    def _enc(self, a, b, rows):
        enc = self.tok([a[i] for i in rows], [b[i] for i in rows], truncation=True, max_length=self.max_len,
                       padding=True, return_tensors="pt")
        return {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}

    def fit(self, a, b, y, epochs=1, batch=64, lr=2e-5, budget_min=120, seed=0):
        from transformers import get_linear_schedule_with_warmup
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * math.ceil(len(y) / batch)
        sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        lossf = torch.nn.BCEWithLogitsLoss()
        y = np.asarray(y, np.float32)
        t0, step = time.time(), 0
        for ep in range(epochs):
            self.model.train()
            perm = np.random.RandomState(seed + ep).permutation(len(y))
            for s in range(0, len(perm), batch):
                rows = perm[s:s + batch]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = self.model(**self._enc(a, b, rows)).logits.squeeze(-1)
                loss = lossf(logits.float(), torch.tensor(y[rows], device=self.device))
                opt.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    sch.step(); step += 1
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); sch.step(); step += 1
                if step % 500 == 0:
                    print(f"[ce] epoch {ep + 1} step {step:,}/{steps:,} loss {loss.item():.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
                if (time.time() - t0) / 60 > budget_min:
                    print("[ce] time budget reached", flush=True)
                    return step
        return step

    @torch.no_grad()
    def predict(self, a, b, batch=512):
        self.model.eval()
        lens = np.fromiter((len(x) + len(z) for x, z in zip(a, b)), np.int32, len(a))
        order = np.argsort(lens, kind="stable")
        out = np.zeros(len(a), np.float32)
        t0 = time.time()
        for s in range(0, len(order), batch):
            rows = order[s:s + batch]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(**self._enc(a, b, rows)).logits.squeeze(-1)
            out[rows] = torch.sigmoid(logits.float()).cpu().numpy()
            if (s // batch) % 400 == 0:
                print(f"[ce] scored {s + len(rows):,}/{len(order):,} ({time.time() - t0:.0f}s)", flush=True)
        return out

    def save(self, d):
        self.model.save_pretrained(d)
        self.tok.save_pretrained(d)


def _pair_text(K, rows, S, tx):
    eid = S.entity_id.astype(str).values
    a = [tx.get(eid[i], "") for i in K.i1.values[rows]]
    b = [tx.get(eid[i], "") for i in K.i2.values[rows]]
    return a, b


def run(paths, tag, model_name, K_tr, S_tr, K_te, S_te, n_train=1_500_000, epochs=1, budget_min=120,
        selftrain=False, unseen_countries=(), seed=0):
    tr = train_rows(paths, K_tr, n_train, seed)
    sc_tr = score_rows(K_tr)
    sc_tr = sc_tr[K_tr.fold.values[sc_tr] == paths.hold]
    sc_te = score_rows(K_te)
    eid_tr, eid_te = S_tr.entity_id.astype(str).values, S_te.entity_id.astype(str).values
    tx_tr = raw_texts(paths, "train", np.r_[eid_tr[K_tr.i1.values[np.r_[tr, sc_tr]]], eid_tr[K_tr.i2.values[np.r_[tr, sc_tr]]]])
    tx_te = raw_texts(paths, "test", np.r_[eid_te[K_te.i1.values[sc_te]], eid_te[K_te.i2.values[sc_te]]])

    mdir = paths.out("train", f"ce_{tag}")
    if os.path.exists(os.path.join(mdir, "config.json")):
        pm = PairModel(mdir)
    else:
        pm = PairModel(model_name)
        a, b = _pair_text(K_tr, tr, S_tr, tx_tr)
        pm.fit(a, b, K_tr.y.values[tr], epochs=epochs, budget_min=budget_min, seed=seed)
        pm.save(mdir)

    out_tr = np.full(len(K_tr), np.nan, np.float32)
    out_tr[sc_tr] = pm.predict(*_pair_text(K_tr, sc_tr, S_tr, tx_tr))
    np.save(paths.out("train", f"score_ce_{tag}.npy"), out_tr)
    out_te = np.full(len(K_te), np.nan, np.float32)
    out_te[sc_te] = pm.predict(*_pair_text(K_te, sc_te, S_te, tx_te))
    np.save(paths.out("test", f"score_ce_{tag}.npy"), out_te)

    if selftrain and len(unseen_countries):
        cty = S_te.country.astype(str).values[K_te.i1.values[sc_te]]
        un = sc_te[np.isin(cty, list(unseen_countries))]
        q, p = out_te[un], K_te.p.values[un]
        pos, neg = un[(q >= 0.97) & (p >= 0.97)], un[(q <= 0.03) & (p <= 0.03)]
        n = int(min(len(pos), len(neg), 100_000))
        if n >= 5_000:
            rng = np.random.RandomState(seed + 1)
            pl = np.r_[rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)]
            rep = rng.choice(tr, min(len(tr), 2 * n), replace=False)
            a1, b1 = _pair_text(K_te, pl, S_te, tx_te)
            a2, b2 = _pair_text(K_tr, rep, S_tr, tx_tr)
            yy = np.r_[np.ones(n), np.zeros(n), K_tr.y.values[rep]]
            pm.fit(a1 + a2, b1 + b2, yy, epochs=1, lr=1e-5, budget_min=budget_min, seed=seed + 2)
            pm.save(paths.out("train", f"ce_{tag}_st"))
            st = np.full(len(K_te), np.nan, np.float32)
            st[un] = pm.predict(*_pair_text(K_te, un, S_te, tx_te))
            np.save(paths.out("test", f"score_ce_{tag}_st.npy"), st)
        else:
            print(f"[ce] self-training skipped: only {n:,} confident pairs per class", flush=True)
    return out_tr, out_te


def existing_scores(paths, split, K):
    """Scores of the pipeline's own cross-encoders, mapped onto K through ce_row."""
    ce_dir = f"{paths.work_dir}/ce"
    files = {"base": "ce_val_scores.parquet", "large": "ce_large_val_scores.parquet"} if split == "train" else \
            {"base": "ce_test_scores.parquet", "large": "ce_large_test_scores.parquet", "base_st": "ce_fr_test_scores.parquet"}
    out = {}
    pos = K.ce_row.values
    for name, f in files.items():
        fp = os.path.join(ce_dir, f)
        if not os.path.exists(fp):
            continue
        s = pd.read_parquet(fp).set_index("row").ce
        v = s.reindex(pos).values.astype(np.float32)
        v[pos < 0] = np.nan
        out[name] = v
        np.save(paths.out(split, f"score_{name}.npy"), v)
    return out
