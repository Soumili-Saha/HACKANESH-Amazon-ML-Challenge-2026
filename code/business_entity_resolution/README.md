# Business Entity Resolution

Matches every Source 1 business record to its Source 2 / Source 3 records and writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv` (in the submission root).

## 1. Layout

```
code/business_entity_resolution/
├── README.md
├── requirements.txt
├── dataset/                 challenge files (train/, test/), not shipped
├── utils/
│   └── validate_submission.py
└── src/
    ├── pipeline/            base pipeline
    │   ├── er_pipeline.py           cleaning, blocking, features, LightGBM, expansion, decision, writer
    │   ├── new_eda.ipynb            data analysis; builds indic_dict.json
    │   ├── indic_dict.json          Indic → English word dictionary learned from train pairs
    │   ├── er_validation_v4.ipynb   train: blocking, stage 1 → expansion → stage 2, 3 S1-grouped folds, final models
    │   ├── test_v4.ipynb            test: blocking, stage 1/2 predictions, candidate export for the cross-encoders
    │   └── ce_colab.ipynb           xlm-roberta cross-encoders (base / large) and unseen-country self-training
    └── erplus/              added models, stacker and decision
        ├── config.py        paths and folds
        ├── context.py       loads the base pipeline's train / test state from its work folder
        ├── dense.py         multilingual-e5-small embeddings, pair cosine, top-k neighbours per country
        ├── candidates.py    candidate table: base candidates + dense-only candidates with stage-1 features and scores
        ├── gbm.py           XGBoost on the stage-1 feature set and dense signals
        ├── cross_encoder.py pair classifier (any Hugging Face model), scoring, unseen-country self-training
        ├── stack.py         LightGBM stacker with per-S1 and per-record competition features
        ├── decide.py        expected-F0.5 decision and rescue threshold tuned on the hold fold
        └── run.py           command-line stages
```

Created when running: `work/` (base pipeline caches), `runs/` (erplus outputs), `../../output/` (submission).

## 2. How the final file is built

```
dataset ─► base pipeline (src/pipeline) ─► candidates (blocking + expansion), LightGBM stage 1/2, xlm-r cross-encoders
                                   │
dense e5 neighbours ───────────────┼─► candidate table (base + dense-only pairs)
                                   │
                    model scores:  LightGBM · dense cosine · XGBoost · xlm-r base / large · multilingual BERT (+ self-trained)
                                   │
                    stacker (hold fold, inner out-of-fold) → p
                                   │
                    unseen-country logit shift → one-to-one + expected-F0.5 set selection → rescue
                                   │
                    output/matching_results.tsv, output/candidate_pairs.tsv
```

## 3. Environment

```bash
python3 -m venv ~/er_venv && source ~/er_venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Paths default to the folders in section 1 and can be overridden with environment variables:

| variable | default | holds |
|---|---|---|
| `ER_DATA_DIR` | `dataset/` | `train/`, `test/` TSVs |
| `ER_PARQUET_DIR` | `dataset/parquet/` | optional parquet copies of the sources and `indic_dict.json` |
| `ER_WORK_DIR` | `work/` | base pipeline caches (`prep/`, `run_frac0.15_seed42/`, `final_v4/`, `test_run_v4/`, `ce/`) |
| `ER_OUT_DIR` | `runs/` | erplus outputs |
| `ER_SUBMISSION_DIR` | `../../output/` | final submission files |

## 4. Run

**Base pipeline** (`src/pipeline/`, notebooks): set the path cell of each notebook to this folder, then run in
order `new_eda.ipynb` → `er_validation_v4.ipynb` → `test_v4.ipynb` (cells 1–9 and the candidate export cell) →
`ce_colab.ipynb`. This fills `work/`.

**Added models** (from `src/`):

```bash
cd src
python -m erplus.run all --subset 2000          # wiring check on 2,000 S1 per side (writes to runs_subset/)

python -m erplus.run prepare                     # candidate table, dense pass, features (train + test)
python -m erplus.run gbm                         # XGBoost
python -m erplus.run ce --tag mbert --model google-bert/bert-base-multilingual-cased --ce_minutes 55 --selftrain
python -m erplus.run stack                       # stacker trained on the hold fold, applied to test
python -m erplus.run decide                      # decision and rescue tuned on the hold fold
python -m erplus.run validate                    # validation table (hold fold)
export ER_BASELINE_OUTPUT=<base pipeline matching_results.tsv>   # kept for test countries absent from training
python -m erplus.run submit                      # writes ../../output/ and runs the validator (only if validation beats the baseline)
```

`ce` can be run again with another `--tag` / `--model` to add more cross-encoders; the stacker uses every
`score_*.npy` present in both `runs/train/` and `runs/test/`.

## 5. Results (hold fold, 111,967 held-out train S1)

| model / step | macro F0.5 |
|---|---|
| LightGBM stage 2 (base pipeline candidates) | 0.987119 |
| LightGBM, + dense candidates | 0.987395 |
| xlm-roberta-base cross-encoder alone | 0.985577 |
| xlm-roberta-large cross-encoder alone | 0.986945 |
| multilingual-BERT cross-encoder alone | 0.988381 |
| XGBoost alone | 0.985203 |
| **stacker + decision (final)** | **0.991776** (India 0.991475, US 0.991982) |

## 6. Validation protocol

Train S1s are split into 3 folds by S1 (seed 42). Fold 2 is the hold fold: it is never used to train the base
models' scores the stacker sees (out-of-fold LightGBM, cross-encoders trained on folds 0–1). The stacker is
trained on hold-fold pairs with an inner 4-way S1 split; the decision and rescue threshold are tuned on the
stacker's out-of-fold predictions. The logit shift applies to test S1s whose country does not occur in training.
