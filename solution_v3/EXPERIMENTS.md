# Experiment log: how the v3 pipeline reached F0.5 0.9880

This log lists every approach tried for the Kaggle GPU pipeline in `solution_v3/`, what each one
measured, and whether it was kept. All scores are the challenge metric (macro F0.5 per Source-1
entity, singletons included) on **held-out Source-1 entities** that the encoder never trained on,
unless marked otherwise.

## Score progression

| Version | Validation F0.5 | Public leaderboard | Test candidates per entity | What changed |
|---|---|---|---|---|
| 1% sample prototype | 0.9936 (sample only) | – | – | whole pipeline built and tested on a 1% sample on CPU |
| **v1** | 0.9856 | – | 10.6 | full-data encoder, multi-view blocking, LightGBM + stage-2 stacker |
| **v2** | 0.9860 | – | 8.1 | name-frequency and competition features, name view for address-less records, hub cap |
| **v3 (submitted)** | **0.9880** | **0.9821** | 8.7 | cleaned-text / legal-form / noise-marker features, 5x more matcher training data, XGBoost on GPU |

Sample scores are not comparable with full-data scores: a 1% pool has far fewer look-alike
entities, so it is much easier.

## Phase 1: pipeline design on a 1% sample (local CPU)

| Experiment | Result | Decision |
|---|---|---|
| Hashed character n-gram bi-encoder (name, consonant skeleton, address), contrastive loss | true entity ranked first for 95.4% of held-out records on the sample | kept: core of the blocking |
| Candidate search with the joint embedding only (top-3) | 96.9% of true pairs kept, 7.9 candidates per entity | replaced |
| **Multi-view search**: joint + address-only + name-only embeddings | 99.1% of true pairs kept with 7.5 candidates per entity; F0.5 ceiling 0.997 | kept |
| Error analysis of blocking misses | almost all misses had an unrelated name (brand names, acronyms like "SHLA") but a near-identical address | motivated the address view |
| Token accounting + house-number distance features against "sibling" decoys (544 vs 557 Portofino Loop, one name word swapped) | sample F0.5 0.9924 -> 0.9936 | kept |
| Stage-2 stacker over competing candidates | slightly worse on the sample (0.9921 vs 0.9924), better on full data | kept (chosen automatically only when it wins) |

## Phase 2: first full-data runs on Kaggle (v1)

| Experiment | Result | Decision |
|---|---|---|
| Encoder on 90% of entities, 4 epochs, hard negatives from epoch 3 | held-out: true entity first 97.5%, within top-20 of some view 99.67% | kept |
| First training attempt | GPU out of memory: sparse gradients stored every padding n-gram slot | fixed by flattening to real n-grams only |
| Exact GPU top-k search | 63 minutes per split (topk over millions of columns) | replaced by a grouped top-k (max of 32-score groups, then top-k inside the best groups), identical results |
| Hub entities | some entities collected up to 8,849 candidates | per-entity cap of 15 per view (cost < 0.0002 of F0.5 ceiling) |
| Policy sweep | 34 policies with polars joins took minutes | vectorised numpy version, all policies in seconds; widened to 216 policies |
| LightGBM 5-fold average applied to 18.4M test pairs | 2.4 hours of CPU prediction | one full model refit on all held-out rows (5x faster) |
| **v1 result** | **0.9856**; 98.9% of true pairs kept; F0.5 ceiling 0.9967 | baseline |

## Phase 3: error analysis and v2

Full-data error analysis of v1 (220K held-out entities, 762K true pairs):

- 3,436 wrong merges vs **24,840 missed pairs**: the loss was recall-side, not precision-side.
- **61% of missed pairs were records without an address.**
- 97.7% of address-less records do have a true match, and when their exact core name belongs to a single
  entity, that entity is the match 97.7% of the time; the model was under-confident on them.

| Experiment | Result | Decision |
|---|---|---|
| Name-frequency features (entities sharing the core name / skeleton) | unique exact-name address-less pairs now predicted correctly (median p 0.996, 99.7% right) | kept |
| Competition features (each similarity minus the best value among the record's other candidates) | part of v2 | kept |
| Name-view search only for address-less records | pair recall 98.9% -> 99.05%, candidates 7.6 -> 6.1 per entity | kept |
| **v2 result** | **0.9860**, 8.1 test candidates per entity | installed as a submission |

## Phase 4: full-scale local experiments (v2 features downloaded from Kaggle)

The 1% sample could not measure these effects, so experiments ran locally on the full feature table
(2.24M candidate pairs, same folds, exact metric).

| Experiment | Result | Decision |
|---|---|---|
| Realistic ceiling: perfect matcher minus unreachable pairs (0.95%) and genuinely ambiguous address-less shared names ("Laex Inc" x6) | **0.9943** | target reference |
| Learning curve (25% / 50% / 100% of training rows) | 0.9831 / 0.9840 / 0.9849: **+0.0009 per doubling** | led to the 50% entity holdout |
| Noise markers (accents, OCR digits, leading junk, duplicated words, domains...) as record-level priors | matched 83% / 82% / 83% / 78% / 96% of the time vs 74% overall | kept as features |
| Cleaned text (PO boxes, phone numbers, URLs, place junk like "ICTY"/"CDP" removed), legal-form overlap/conflict, noise markers | stage 1 **0.9849 -> 0.9865 (+0.0016)** | kept |
| Per-entity assignment features (other records' probabilities by source) | +0.0002 | not adopted |

## Phase 5: v3 on Kaggle

| Step | Result | Decision |
|---|---|---|
| Encoder trained on 50% of entities so the matcher learns from 5x more unseen entities | true entity first 96.9% (vs 97.5%); F0.5 ceiling 0.9970 (vs 0.9971) | kept |
| XGBoost on GPU instead of CPU LightGBM | equal quality locally (0.98666 vs 0.98665), minutes instead of hours on 5x data | kept |
| Attempt 1 | CPU out of memory building 9.5M x 130 training features | features built in record-aligned chunks written to parquet |
| Attempt 2 | stopped at start (syntax error in the job script) | job scripts are compile-checked before every push |
| Attempt 3 | GPU out of memory in XGBoost (PyTorch cache + per-fold MLP) | clear the GPU cache before the matcher; MLP off (never won the blend) |
| Crash-proof job script | Kaggle discards outputs of errored jobs; the script now always exits cleanly | the resume job reused the saved encoder, policy and searches |
| **v3 result** | **0.9880** (India 0.9877, US 0.9882, singletons 0.9852); leaderboard **0.9821** | submitted |

## Phase 6: singletons and final tuning (local, full scale)

Where v2's remaining loss came from (220K entities):

| Case | Share of lost F0.5 |
|---|---|
| Entities with some missed matches | 51.6% |
| Entities with some wrong merges | 22.8% |
| Entities with matches predicted empty | 15.8% |
| Singletons given a false match | 9.4% |

| Experiment | Result | Decision |
|---|---|---|
| Entity-level gate: P(entity has any match) from all its candidates | AUC 0.9994 but F0.5 +0.00001 | not adopted |
| Specialist model for entities with exactly one strong candidate | +0.00014 (noise level) | not adopted |
| Bigger trees (LightGBM 255 leaves), depth-10 XGBoost | 0.98666 for every variant | not adopted |
| Averages of 2-4 tree models | at most +0.00015 | not adopted |

The remaining singleton errors are decoy records that are near-exact noisy copies of the singleton
entity (89.5% of singleton false matches) and address-less records with shared names; the text
does not contain the information to separate them.

## Why the leaderboard is below validation

Validation 0.9880 vs public leaderboard 0.9821. Two shifts the validation cannot measure:

- **France** (15% of test entities, absent from training) has very generic, repetitive names ("Ecole",
  "Lycée", "Amicale", "Centre Hospitalier"). If US and India score at their validation level, France is
  around 0.95.
- **Denser test set**: 5.75 Source-2/3 records per Source-1 entity vs 4.7 in training, so more decoys
  per entity.

Next steps that target this gap: validate with extra decoys to match the test density and re-tune
the decision rule; treat generic French institution words as low-weight filler; save test
probabilities so decision-rule changes can be tried without a full rerun.

## What moved the score, in order of impact

1. Multi-view blocking (address view, then the address-less name view): the recall ceiling.
2. Understanding the noise process: cleaned text, legal forms, noise markers (+0.0016).
3. Name uniqueness for address-less records (v2).
4. More unseen training entities for the matcher (50% holdout, +0.0009 per doubling).
5. Engineering that made full-scale runs possible on free GPUs: flat embedding bags, grouped top-k,
   chunked features, resumable and crash-proof Kaggle jobs.
