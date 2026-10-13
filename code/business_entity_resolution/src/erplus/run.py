import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from . import candidates, cross_encoder, decide, dense, gbm, stack
from .config import Paths, load_er
from .context import light_S, load_context


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def unseen(paths):
    tr = set(light_S(paths, "train").country.astype(str).unique())
    te = set(light_S(paths, "test").country.astype(str).unique())
    return sorted(te - tr)


def pick_subset(ctx, paths, n, seed=0):
    rng = np.random.RandomState(seed)
    s1 = np.flatnonzero((ctx.S.src == "S1").values)
    if ctx.split == "train":
        f = ctx.fold[s1]
        h = rng.choice(s1[f == paths.hold], min(n, int((f == paths.hold).sum())), replace=False)
        t = rng.choice(s1[f != paths.hold], min(n, int((f != paths.hold).sum())), replace=False)
        keep = np.sort(np.r_[h, t])
    else:
        keep = np.sort(rng.choice(s1, min(n, len(s1)), replace=False))
    np.save(paths.out(ctx.split, "subset_s1.npy"), keep)
    return keep


def stage_prepare(paths, er, split, args):
    ctx = load_context(paths, er, split, n_workers=args.workers)
    log(f"{split}: {len(ctx.CE):,} candidates, {int((ctx.S.src == 'S1').sum()):,} S1")
    keep = pick_subset(ctx, paths, args.subset) if args.subset else None
    if keep is not None:
        cand_i2 = ctx.CE.i2.values[np.isin(ctx.CE.i1.values, keep)]
        pool = np.flatnonzero((ctx.S.src != "S1").values)
        extra = np.random.RandomState(1).choice(pool, min(len(pool), 20 * len(keep)), replace=False)
        rows = np.unique(np.r_[keep, cand_i2, extra])
        emb = dense.embeddings(paths, split, ctx.S, rows=rows)
        knn = dense.knn_pairs(emb, ctx.S, k=args.knn_k, query_rows=keep, pool_rows=rows)
    else:
        emb = dense.embeddings(paths, split, ctx.S)
        knn = dense.knn_pairs(emb, ctx.S, k=args.knn_k)
    D = dense.new_pairs(knn, ctx.CE, k_new=args.knn_new)
    del knn
    log(f"{split}: dense neighbours {len(D):,} new pairs")
    K, rows, X = candidates.build(paths, er, ctx, D, emb, s1_keep=keep, n_workers=args.workers)
    log(f"{split}: candidate table {len(K):,} rows ({int((K.origin.values == candidates.ORIGIN_DENSE).sum()):,} dense), "
        f"gbm rows {len(rows):,}")
    cross_encoder.existing_scores(paths, split, K)


def stage_gbm(paths, er, args):
    Ktr, rtr, Xtr = candidates.load(paths, "train")
    if os.path.exists(paths.out("train", "xgb.json")) and os.path.exists(paths.out("train", "xgb_info.json")):
        import xgboost as xgb
        m = xgb.Booster()
        m.load_model(paths.out("train", "xgb.json"))
    else:
        m = gbm.fit(paths, Ktr, rtr, Xtr, device=args.device)
    gbm.score(paths, "train", Ktr, rtr, Xtr, m)
    Kte, rte, Xte = candidates.load(paths, "test")
    gbm.score(paths, "test", Kte, rte, Xte, m)
    log("gbm scores written")


def stage_ce(paths, er, args):
    Ktr, _, _ = candidates.load(paths, "train")
    Kte, _, _ = candidates.load(paths, "test")
    cross_encoder.run(paths, args.tag, args.model, Ktr, light_S(paths, "train"), Kte, light_S(paths, "test"),
                      n_train=args.ce_train, epochs=args.ce_epochs, budget_min=args.ce_minutes,
                      selftrain=args.selftrain, unseen_countries=unseen(paths))
    log(f"cross-encoder {args.tag} scores written")


def stage_stack(paths, er, args):
    un = unseen(paths)
    S_tr, S_te = light_S(paths, "train"), light_S(paths, "test")
    Ktr, _, _ = candidates.load(paths, "train")
    p_tr, _ = stack.fit(paths, Ktr, S_tr, stack.load_sources(paths, "train", Ktr))
    Kte, _, _ = candidates.load(paths, "test")
    stack.apply(paths, Kte, S_te, stack.load_sources(paths, "test", Kte, S_te, un))
    log("stacker trained and applied")


def stage_decide(paths, er, args):
    S = light_S(paths, "train")
    gt = er.load_ground_truth(paths.data_dir, paths.parquet_dir)
    fold = er.s1_folds(S, paths.n_folds, paths.seed)
    K, _, _ = candidates.load(paths, "train")
    p = np.load(paths.out("train", "p_stack.npy"))
    rep = decide.tune(er, paths, K, S, gt, fold, p)
    log(f"hold F0.5 {rep['F05_hold']:.6f} (pipeline baseline {rep['F05_hold_baseline_lgbm']:.6f}) | "
        f"decision {rep['decision']} rescue {rep['rescue']}")


def stage_validate(paths, er, args):
    S = light_S(paths, "train")
    gt = er.load_ground_truth(paths.data_dir, paths.parquet_dir)
    fold = er.s1_folds(S, paths.n_folds, paths.seed)
    K, _, _ = candidates.load(paths, "train")
    p = np.load(paths.out("train", "p_stack.npy"))
    rep = decide.report(er, paths, K, S, gt, fold, stack.load_sources(paths, "train", K), p)
    log(f"validation on {rep['n_hold_s1']:,} held-out train S1 (macro F0.5):")
    for r in rep["rows"]:
        log(f"  {r['F05_hold']:.6f}  {r['model']}")
    log(f"  by country: { {k: round(v, 6) for k, v in rep['final_by_country'].items()} }")
    log(f"  gain over pipeline baseline: {rep['final'] - rep['baseline']:+.6f}")


def stage_submit(paths, er, args):
    vf = paths.out("train", "validation.json")
    if not args.force:
        assert os.path.exists(vf), "run the validate stage first"
        v = json.load(open(vf))
        assert v["final"] > v["baseline"], (f"validation {v['final']:.6f} does not beat the pipeline baseline "
                                            f"{v['baseline']:.6f}; use --force to write the test output anyway")
    S = light_S(paths, "test")
    K, _, _ = candidates.load(paths, "test")
    p = np.load(paths.out("test", "p_stack.npy"))
    out_dir = args.submission_dir or (paths.out("test", "submission") if args.subset else paths.submission_dir)
    base = None if args.no_baseline_unseen else args.baseline_output
    (mp, cp), keep = decide.finalize(er, paths, K, S, p, unseen(paths), delta=args.shift, out_dir=out_dir,
                                     baseline_path=base if base and os.path.exists(base) else None)
    log(f"submission: {mp} | {cp} | matched pairs {int(keep.sum()):,}")
    validator = os.path.join(os.path.dirname(paths.code_dir.rstrip("/")), "..", "utils", "validate_submission.py")
    if os.path.exists(validator) and not args.subset:
        subprocess.run([sys.executable, validator, "--matching", mp, "--candidate", cp,
                        "--test-dir", os.path.join(paths.data_dir, "test")])


STAGES = ["prepare", "gbm", "ce", "stack", "decide", "validate", "submit"]


def main():
    ap = argparse.ArgumentParser(prog="python -m erplus.run")
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--subset", type=int, default=0, help="run on N S1 records per side (wiring check)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--knn_k", type=int, default=20)
    ap.add_argument("--knn_new", type=int, default=10)
    ap.add_argument("--tag", default="mbert")
    ap.add_argument("--model", default="google-bert/bert-base-multilingual-cased")
    ap.add_argument("--ce_train", type=int, default=1_500_000)
    ap.add_argument("--ce_epochs", type=int, default=1)
    ap.add_argument("--ce_minutes", type=float, default=120)
    ap.add_argument("--selftrain", action="store_true")
    ap.add_argument("--shift", type=float, default=-1.0, help="logit shift for test countries absent from training")
    ap.add_argument("--submission_dir", default=None)
    ap.add_argument("--skip_ce", action="store_true")
    ap.add_argument("--baseline_output", default=os.environ.get("ER_BASELINE_OUTPUT", ""),
                    help="pipeline submission file; its matches are kept for test countries absent from training")
    ap.add_argument("--no_baseline_unseen", action="store_true")
    ap.add_argument("--force", action="store_true", help="write the test output even if validation does not beat the baseline")
    args = ap.parse_args()

    paths = Paths()
    if args.subset:
        paths.out_dir = paths.out_dir.rstrip("/") + "_subset"
        args.ce_train, args.ce_minutes = min(args.ce_train, 20_000), min(args.ce_minutes, 10)
    er = load_er(paths)
    splits = ["train", "test"] if args.split == "both" else [args.split]
    todo = STAGES if args.stage == "all" else [args.stage]
    for st in todo:
        t0 = time.time()
        log(f"== {st}")
        if st == "prepare":
            for sp in splits:
                stage_prepare(paths, er, sp, args)
        elif st == "ce":
            if not args.skip_ce:
                stage_ce(paths, er, args)
        else:
            globals()[f"stage_{st}"](paths, er, args)
        log(f"== {st} done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
