# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** HACKANESH  
**Team Members:** Soumili Saha (team leader), Soumyajit Pramanik, Supratik Das, Sougata Daw  
**Submission Date:** 2 October 2026

---

## 1. Executive Summary
We clean names and addresses in a country-agnostic way, generate candidates with location partitions, exact
keys, character TF-IDF neighbours, expansion from confident matches and a multilingual dense-embedding pass,
and score pairs with a two-stage LightGBM, an XGBoost model and transformer cross-encoders. A LightGBM stacker
combines every score with per-S1 and per-record competition features, and an expected-F0.5 decision under a
one-to-one constraint picks the final matches, followed by a tuned rescue step.

---

## 2. Methodology

### 2.1 Problem Analysis
These findings come from our exploratory data analysis. The full analysis, with the tables, examples and
measurements behind every point, is in
**`code/business_entity_resolution/src/pipeline/new_eda.ipynb`**.


**Size and structure**
- **Size:** train S1 2.2M, S2 5.0M, S3 5.3M; test S1 1.7M, S2 4.9M, S3 5.1M. Even inside one country there
  are trillions of possible pairs (US alone ≈ 1.32M × 6.19M ≈ 8 trillion), so blocking is required.
- **Country is a perfect hard block:** a true match never crosses country labels, so records are only compared
  within the same country label, whatever labels appear.
- **Several matches per S1:** only 5.6 % of S1 records have no match; most have 2–6 (≈ 3.5 on average, about
  half from S2 and half from S3). Each source holds several copies of the same business.
- **One-to-one:** every S2/S3 record belongs to at most one S1 (0 violations). At prediction time a record that
  fits two S1s is given only to the best one, a strong precision tool for F0.5.
- **Look-alikes:** ≈ 26 % of S2/S3 records (≈ 1.3M per source) match nothing. They are clean records of other,
  similar businesses and are the main source of false matches.
- **Test differs from train:** test contains a new test country with no training labels (≈ 15 % of test
  S1s, ≈ 14.5 % of test S2/S3) and new vocabulary (retail words such as stores, jewellers, pharmacy that never
  appear in train). Everything must therefore be country- and vocabulary-agnostic.

**Where the noise is**
- **S1 is clean, S2/S3 carry the noise:** S1 has no native script, junk, domain names or all-caps names. The task
  is to bring S2/S3 back to S1's form.
- **Noise was added to matched records, not unmatched ones:** e.g. empty address 4–5 % (matched) vs 0.2–0.4 %
  (unmatched), domain as name 4.5–5.9 % vs 0.5–0.7 %. So the amount of noise is itself a useful signal, and
  blocking needs a name-only path (≈ 5 % of matches have no address).
- **Postal codes are almost never present:** US zip in ≈ 10 % of records, India PIN in ≈ 1 %, the new test
  country ≈ 0.4 %. They can only be a bonus, not a blocking key.
- **Native script is a major gap for India:** ≈ 23 % of India S2 names (≈ 13 % of S3) and ≈ 23 % of addresses are
  in Indic scripts; without transliteration only 6.4 % of these pairs share a word. A dictionary learned from
  train pairs (1,302 words, `indic_dict.json`) recovers them, with a rule-based skeleton fallback (83–86 %) for
  unseen words.

**Name noise**
- Legal suffixes dropped, added, moved or swapped between types (pc→inc, lp→llc, limited→ltd), including
  dotted forms (L.L.C.) and native-script forms.
- Injected filler words (center, services, partners; location prefixes such as downtown, riverside), found
  automatically: a word far more common in S2/S3 than in S1 is an injected filler. This works for any language.
- OCR-style swaps (hea1th, 8low, rn↔m, i↔l), heavy multi-letter typos, injected accents, shuffled or dropped
  words, honorifics (M/s, Sri, Mr), alias markers (dba, aka, formerly).
- Glued / domain-style names (parkerrestaurant.com) and fully random names built from a fixed set of syllables
  (Fayepyra, Zetavio). No name rule can catch those; only the address can.

**Address noise**
- Abbreviations in both directions (Road↔Rd, Street↔St↔Saint), numbers as words, state code vs. full name vs.
  native script, city aliases (Bombay↔Mumbai, Gurgaon↔Gurugram), reordered components, junk (PO Box, PMB, null).
- House numbers corrupted in a specific way, mostly one digit deleted or changed (707→70, 814→813), so exact
  number keys are unsafe and a "delete one digit" key is used.
- Only ≈ 13 % of true address pairs have the same word set, so blocking uses address parts (state, locality,
  numbers), not the whole address. State agrees in 99.8 %+ of pairs and is a safe partition for the US and India.

**What this meant for the design**
- Blocking needs two independent paths, name and address, because some matches have a random name but the
  right address and others the right name but no address.
- Names in the new test country and Indic names are built from a small common vocabulary, so a single name
  word is never a good key; keys must combine name and location.
- Exact keys give high recall but far too many candidates (≈ 280 per S1), so candidates are ranked by a
  similarity score and only the top K per S1 are kept.
- The hardest negatives are near-duplicates (same name and street, different house number) and true copies of
  another S1.


### 2.2 Solution Strategy

```
records ─► cleaning + transliteration (indic_dict + phonetic fallback)
        ─► candidates: location partitions · exact keys · char TF-IDF kNN · expansion · dense e5 kNN
        ─► scores: LightGBM stage 1/2 · dense cosine · XGBoost · xlm-r base / large · multilingual BERT (+ self-trained)
        ─► stacker (competition features) ─► unseen-country shift ─► one-to-one + expected-F0.5 ─► rescue
```

**Approach Type:** Hybrid: blocking + gradient-boosted classifiers + transformer cross-encoders + stacking.  
**Core Innovation:**
- Label-free adaptation to unseen data: filler words detected from S2/S3-vs-S1 frequency, synthetic-name
  detection from the data's own syllable inventory, transliteration with a learned dictionary and a phonetic
  fallback for unseen words.
- Candidate expansion from confident matches (siblings share exact name / glued-name / full-address keys)
  combined with a dense multilingual pass, so both structural and semantic duplicates are found.
- A stacker that sees every model score in competition (rank inside the S1's candidates, how many S1s want the
  same record) and a decision that maximises expected F0.5 directly.

---

## 3. Candidate Generation (Blocking)

| step | method |
|---|---|
| partitions | country → state when a state can be detected; records without a usable address go to a per-country no-address pool |
| exact keys | shared name / address keys inside the partition (keys covering > 200 records skipped) |
| char TF-IDF kNN | character 3-gram name vectors + address-token vectors |
| rank and cut | keys ∪ kNN ranked by name + address similarity |
| expansion | after stage 1, confident matches seed exact name / glued-name / full-address keys |
| dense pass | multilingual-e5-small embeddings of the cleaned `name \| address`, top-k neighbours per country |

- **Blocking keys used:** location partitions (country, state), exact name-set / glued-name / full-address keys,
  character 3-gram TF-IDF on names, token TF-IDF on addresses, dense multilingual embeddings.
- **Candidate pairs generated:** 68,813,644 test pairs in `candidate_pairs.tsv` (39.7 per S1): 62,888,529 from
  partitions, keys, TF-IDF and expansion, plus 5,925,115 from the dense pass. On train, the dense pass adds
  1,052,667 pairs (after removing pairs whose S2/S3 record belongs to an S1 outside the selection).
- **How you ensured true matches were not lost:** recall is measured on held-out train S1s against all their
  candidates: 98.5 % before expansion, 98.8 % with expansion (≈ 31 candidates per S1). The dense pass targets the
  remaining misses (typos, glued names, native script); with LightGBM alone it raises hold-fold F0.5 from
  0.987119 to 0.987395, and the stacker uses it as an extra signal (section 5).

---

## 4. Matching Model

**Features used:**
- **Name features:** TF-IDF similarity, token overlap / Jaccard / containment, exact-set and glued-name equality,
  legal-form presence and conflicts, filler words, synthetic-name share, split-token similarity.
- **Address features:** token similarity, exact address, house-number and number-set agreement / conflict,
  no-address flags.
- **Other:** blocking and expansion flags, dense cosine and neighbour rank, sibling context (the S1's other
  confident candidates), competition for the same S2/S3 record, source (S2 / S3).

**Model type:**

| model | role |
|---|---|
| LightGBM stage 1 → stage 2 | pair classifier; stage 2 adds sibling context; 3 S1-grouped folds, early stopping |
| XGBoost | second gradient-boosting model on the stage-1 features + dense signals |
| xlm-roberta-base / large cross-encoders | read both records together; contested pairs |
| multilingual-BERT (bert-base-multilingual-cased) cross-encoder | different model family for diversity; self-training on unseen countries |
| LightGBM stacker | all scores + per-S1 rank / gap / second best + per-record competition |

**Threshold selection method:** each S2/S3 record is kept only for its best S1; per S1, the prefix of candidates
with the highest expected F0.5 is kept (empty allowed), with γ / λ / miss tuned on the hold fold; a rescue step
re-adds unassigned records whose stacker probability exceeds a threshold tuned on the hold fold. Test S1s whose
country does not occur in training get a logit shift before the decision.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.991776** on the hold fold (111,967 held-out train S1s; India 0.991475, US 0.991982),
  against 0.987119 for the base pipeline's LightGBM on the same S1s (+0.004658).
- **Common false positives (wrong merges):** near-duplicates with the same name and street but a different house
  number; generic names in another city; a different business at the same address; records with an empty
  address.
- **Common false negatives (missed matches):** house numbers changed on the true match, different trade names at
  the same address, empty addresses with noisy names, native-script names.

---

## 6. Conclusion
Country-agnostic cleaning with label-free statistics, structural plus dense candidate generation, several
complementary scorers and a competition-aware stacker with an expected-F0.5 decision form the solution. The main
lessons: measure blocking recall on held-out S1s, use the one-to-one structure in the decision, and adapt to
unseen countries with label-free signals rather than hand-made lists.

---

## Appendix

### A. Code Artefacts
All code is in `code/business_entity_resolution/`; its `README.md` lists every command.
- `src/pipeline/`: base pipeline (`er_pipeline.py` + notebooks `new_eda` → `er_validation_v4` → `test_v4` → `ce_colab`).
- `src/erplus/`: dense pass, XGBoost, cross-encoders, stacker, decision; `python -m erplus.run <stage>` from `src/`.
- `utils/validate_submission.py`: output check.

**Compute:** AWS EC2 g5.2xlarge (NVIDIA A10G) and Google Colab (GPU and TPU).

### B. Additional Results
Hold fold (111,967 held-out train S1s), macro F0.5 with the tuned expected-F0.5 decision:

| model / step | F0.5 |
|---|---|
| LightGBM stage 2 (base pipeline candidates) | 0.987119 |
| LightGBM, + dense candidates | 0.987395 |
| xlm-roberta-base cross-encoder alone | 0.985577 |
| xlm-roberta-large cross-encoder alone | 0.986945 |
| multilingual-BERT cross-encoder alone | 0.988381 |
| XGBoost alone | 0.985203 |
| **stacker + decision (final)** | **0.991776** |

Final decision: expected F0.5 with γ = 1.0, λ = 0.1, miss = 0.25; the rescue step did not improve the hold fold
and is off. For test S1s whose country does not occur in training, the base pipeline's matches are kept.
Test output: 5,782,445 matched pairs, 94.1 % of S1s with at least one match.
