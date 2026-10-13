import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CAND_COLS = ["i1", "i2", "bits", "knn", "sn", "sa", "pool", "exp_sib", "exp_kaddr"]


@dataclass
class Context:
    split: str
    S: pd.DataFrame
    C: pd.DataFrame
    CE: pd.DataFrame
    p: np.ndarray
    n_blk: int
    res: object = None
    gt: pd.DataFrame = None
    y: np.ndarray = None
    fold: np.ndarray = None
    extra: dict = field(default_factory=dict)


def light_S(paths, split):
    d = paths.run_dir if split == "train" else paths.test_dir
    S = pd.read_parquet(f"{d}/selected.parquet")
    for c in ("src", "country"):
        S[c] = S[c].astype("category")
    return S


def train_context(paths, er, n_workers=4):
    fillers = json.load(open(f"{paths.work_dir}/fillers_train.json"))
    S = light_S(paths, "train")
    parts = json.load(open(f"{paths.run_dir}/partitions.json"))
    S = er.attach_full_tokens(S, "train", paths.work_dir)
    S, _ = er.setup_v3_resources(S, "train", paths.data_dir, paths.work_dir, paths.parquet_dir)
    C = er.run_blocking(S, parts, paths.run_dir, n_workers=n_workers)
    res = er.build_resources(S, fillers)
    gt = er.load_ground_truth(paths.data_dir, paths.parquet_dir)
    Cm = C[er.realistic_mask(C, S, gt)].reset_index(drop=True)
    del C
    E = pd.read_parquet(f"{paths.v4_dir}/E.parquet")
    CE = er.concat_candidates(Cm, E)
    del E
    p = np.load(f"{paths.v4_dir}/p_final.npy").astype(np.float32)
    assert len(p) == len(CE), "p_final.npy does not match the candidate table"
    y = er.label_pairs(CE, S, gt).astype(np.int8)
    fold = er.s1_folds(S, paths.n_folds, paths.seed)
    return Context("train", S, None, CE, p, len(Cm), res=res, gt=gt, y=y, fold=fold)


def test_context(paths, er, n_workers=4):
    import lightgbm as lgb
    met = json.load(open(f"{paths.final_dir}/metrics_v4.json"))
    fillers = json.load(open(f"{paths.test_dir}/fillers_test.json"))
    S = light_S(paths, "test")
    parts = json.load(open(f"{paths.test_dir}/partitions.json"))
    syl = json.load(open(f"{paths.final_dir}/syllables_train.json"))
    S, _ = er.setup_v3_resources(S, "test", paths.data_dir, paths.work_dir, paths.parquet_dir, extra_syllables=syl)
    C = er.run_blocking(S, parts, paths.test_dir, n_workers=n_workers)
    res = er.build_resources(S, fillers)
    m1 = lgb.Booster(model_file=f"{paths.final_dir}/model_stage1.txt")
    m2 = lgb.Booster(model_file=f"{paths.final_dir}/model_stage2.txt")
    p1 = er.predict_stage1_streaming(C, S, res, m1, paths.test_dir, n_workers=n_workers)
    E = pd.read_parquet(f"{paths.test_dir}/E.parquet")
    p1E = np.load(f"{paths.test_dir}/p1E.npy")
    CE = er.concat_candidates(C, E)
    n_c = len(C)
    p1_all = np.r_[p1, p1E].astype(np.float32)
    del C, E, p1, p1E
    p = er.predict_stage2(CE, S, res, p1_all, m2, paths.test_dir, min_p=met["min_p"], n_c=n_c, n_workers=n_workers)
    del p1_all
    return Context("test", S, None, CE, np.asarray(p, np.float32), n_c, res=res,
                   extra={"m1": m1, "decision": met["decision"]})


def load_context(paths, er, split, n_workers=4):
    return train_context(paths, er, n_workers) if split == "train" else test_context(paths, er, n_workers)


def train_countries(paths):
    return set(light_S(paths, "train").country.astype(str).unique())
