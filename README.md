<div align="center">

# Business Entity Resolution

**Amazon ML Challenge 2026 · Team Gradient_Descendants**

Match every Source-1 business to all Source-2 / Source-3 records of the same real-world entity.

![Best validation F0.5](https://img.shields.io/badge/validation%20F0.5-0.9880-2ea44f)
![Public leaderboard](https://img.shields.io/badge/leaderboard-0.9821-blue)
![Candidates per entity](https://img.shields.io/badge/test%20candidates%20%2F%20S1-8.7-orange)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![No pretrained weights](https://img.shields.io/badge/models-trained%20from%20scratch-lightgrey)

</div>

---

## Contents

- [The problem](#the-problem)
- [Results at a glance](#results-at-a-glance)
- [Approach 1: bi-encoder blocking + XGBoost matcher (best, submitted)](#approach-1-bi-encoder-blocking--xgboost-matcher)
- [Approach 2: TF-IDF retrieval + LightGBM on free-tier EC2](#approach-2-tf-idf-retrieval--lightgbm-on-free-tier-ec2)
- [Approach 3: TF-IDF retrieval + learned candidate pruner](#approach-3-tf-idf-retrieval--learned-candidate-pruner)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Licences and rules](#licences-and-rules)

---

## The problem

Three sources of business records (`entity_id, business_name, business_address, country`):
**S1** is a deduplicated reference; **S2** and **S3** hold noisy copies (typos, transliterations,
filler words, missing addresses, near-copy decoys). For every S1 entity, predict all S2/S3 records
that describe the same business: zero, one or many.

| | S1 | S2 | S3 | Countries |
|---|---|---|---|---|
| Train | 2.21M | 5.03M | 5.29M | US, India |
| Test | 1.73M | 4.89M | 5.08M | US, India, **France (unseen in training)** |

- **Metric:** F0.5 per S1 entity, macro-averaged over *all* S1 entities (precision counts twice as
  much as recall; for a singleton, an empty prediction scores 1 and anything else scores 0).
- **Submission:** `matching_results.tsv` plus `candidate_pairs.tsv`, which must contain exactly the
  pairs the final model scored. Smaller candidate sets per S1 entity rank higher.

---

## Results at a glance

Approaches are ordered by F0.5, best first.

| # | Approach | Validation macro F0.5 | Leaderboard Score | Candidates per S1 | Compute |
|:-:|---|:-:|:-:|:-:|---|
| 🥇 **1** | **Bi-encoder blocking + XGBoost matcher** (`solution_v3/`) | **0.9880** | **0.9821** | **8.7** (test) | free Kaggle T4 GPU, ~3.5 h |
| 🥈 2 | TF-IDF retrieval + LightGBM (notebooks, `src/`) | 0.9740 | **0.977** | 70.4 | free-tier EC2 CPU, 2 vCPU / 8 GB |
| 🥉 3 | TF-IDF retrieval + learned candidate pruner + LightGBM | **0.9662** | **0.966** | 7.5 | free-tier EC2 CPU |

¹ Measured on the full-scale India block only. On the same block, Approach 2 without the pruner scores
0.9698, so the pruner cuts candidates 70.4 → 7.5 per S1 for a loss of only 0.0002 F0.5.

---

## Approach 1: bi-encoder blocking + XGBoost matcher

> **Best result, submitted.** Validation F0.5 **0.9880** on 1.1M held-out S1 entities · public
> leaderboard **0.9821** · 8.7 test candidates per S1 · 99.0% of true pairs kept by blocking.
> Code and docs: [`solution_v3/`](solution_v3/)

We treat entity resolution as *"which S1 entity does each S2/S3 record belong to, if any?"*.

```
raw TSVs ─► normalise + clean ─► char n-gram bi-encoder ─► multi-view GPU kNN ─► ~150 pair features
        ─► XGBoost (GPU) + stage-2 stacker ─► one entity per record ─► expected-F0.5 decision
```

| Stage | What it does |
|---|---|
| **Normalise** | ASCII transliteration, legal-form sets, cleaned copies without filler words, phone numbers, PO boxes, URLs; 12 record-level **noise markers** |
| **Bi-encoder** | Hashed character 2–5-gram encoder trained from scratch (InfoNCE, mined hard negatives). Outputs a joint, a name-only and an address-only embedding |
| **Blocking** | Exact GPU top-k per country from the record side. Joint top-5, address top-2 (cos > 0.8) to rescue brand-name / acronym records, name top-5 only for address-less records, cap of 15 per S1 per view. The policy is picked from 216 candidates on held-out entities |
| **Features** | Embedding scores and ranks, rapidfuzz name/address similarities, token accounting, house-number distance, legal form, noise markers, **name frequency**, **competition** against the record's other candidates |
| **Matcher** | XGBoost on GPU (LightGBM on CPU gives the same score), 5-fold on held-out entities, plus a stage-2 stacker over competing candidates |
| **Decision** | Each record keeps its best S1; each S1 keeps the prefix that maximises expected F0.5, or stays empty when P(no match) wins |

**Key idea for validation:** 50% of S1 entities are held out. The encoder trains on the other half,
and the matcher trains only on candidates of entities the encoder never saw, so validation scores
reflect test behaviour.

**Score progression**

| Version | Change | Validation F0.5 | Test cand / S1 |
|---|---|:-:|:-:|
| v1 | bi-encoder, multi-view blocking, LightGBM + stacker | 0.9856 | 10.6 |
| v2 | name-frequency and competition features, name view for address-less records | 0.9860 | 8.1 |
| **v3** | cleaned-text / legal-form / noise features, 5× more matcher data, XGBoost on GPU | **0.9880** | 8.7 |

<details>
<summary><b>Run it</b></summary>

```bash
cd solution_v3/src
pip install -r ../requirements.txt
export BER_HOLDOUT_PCT=50 BER_MLP=0 BER_GB_VARIANTS=lg127     # settings of the submitted v3 run
python pipeline.py run-all --data <dataset dir> --work ../work --models ../models --out <output dir>
```

A GPU is strongly recommended. Every stage is checkpointed and resumes when re-run. A 1% smoke test on
CPU and the exact Kaggle jobs are described in [`solution_v3/README.md`](solution_v3/README.md).

</details>

---

## Approach 2: TF-IDF retrieval + LightGBM on free-tier EC2

> Validation F0.5 **0.9740** (valid-B, full data) · P 0.9934 / R 0.9436 · ~70 candidates per S1.
> Code: [`src/ber/`](src/ber/) and [`notebooks/`](notebooks/)

Built to run on the **full dataset on an AWS free-tier instance** (m7i-flex.large: 2 vCPU,
8 GB RAM + 4 GB swap, 30 GB gp3). Every stage streams or chunks.

```
raw TSVs (S3) ─► 00 clean ─► 01 EDA ─► 02 block + TF-IDF top-K ─► 03 features ─► 04 LightGBM + threshold ─► 05 test inference
```

| Stage | Method |
|---|---|
| **00 preprocessing** | Streamed; ASCII transliteration (anyascii), legal-form canonicalisation, alias (fka/dba) split, address normalisation, **phonetic consonant skeletons** of name and address |
| **02 candidates** | Hard block on `country`; sparse TF-IDF top-15 from each S2/S3 record into S1 on name char-3grams ⊕ phonetic-name 3grams ⊕ address word 1-2grams ⊕ phonetic-address 1-2grams |
| **03 features** | ~70 country-agnostic features: rapidfuzz similarities, TF-IDF cosines, numeric-token agreement, legal form, **competition** (rank, margin to best rival), **ambiguity** (how many S1 share a name / address) |
| **04 model** | LightGBM on an entity-disjoint subsample: 30% of S1 entities to train, 2% valid-A for early stopping and threshold, 2% valid-B untouched for reporting |
| **05 inference** | ~150M test pairs featurised and scored in a stream; single threshold |

<details>
<summary><b>Run it</b></summary>

```bash
python3 -m venv ~/ber-venv && source ~/ber-venv/bin/activate
pip install -r requirements.txt
export BER_S3_BUCKET=<bucket>          # or "none" and set BER_RAW_DIR to a local folder with train/ and test/
# export BER_SAMPLE_FRAC=0.02          # optional fast dev run on a 2% sub-sample
tmux new -d -s run "bash scripts/run_notebooks.sh > ~/run.log 2>&1"     # runs notebooks 00 → 05

python scripts/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir <folder with test_source1.tsv>
```

Rough timings on the free-tier instance: cleaning ~5 min per split, retrieval ~1.5 h per split,
training features ~30 min, model ~15 min, streamed test scoring ~2–3 h. Large intermediates stay on
EBS; only raw data, model and submission go to S3 (inside the S3 free tier).

</details>

---

## Approach 3: TF-IDF retrieval + learned candidate pruner

> Validation F0.5 **0.9697** (India block, full scale) · **7.5** candidates per S1 (down from 70.4).
> Code: `pipeline.train_pruner`, `pipeline.run_prune` in [`src/ber/`](src/ber/); notebooks 02 (section 4) and 05 (section 2b)

The organisers announced that smaller candidate sets rank higher. This approach adds a **cascade**
stage to Approach 2: a small LightGBM that sees only retrieval-stage signals (score split into
name/address parts, rank, competition, near-ties, name/address frequency) and drops pairs with
q < `PRUNE_Q`. The survivors are the final `candidate_pairs.tsv` and the only pairs the full matcher
scores, so test scoring drops from ~2.5 h to ~20 min.

| Candidate stage (India block) | Cand / S1 | Candidate recall | Valid-B F0.5 |
|---|:-:|:-:|:-:|
| Retrieval top-15 (Approach 2) | 70.4 | 0.9753 | 0.96983 |
| Rank ≤ 2 rule | 9.1 | 0.9547 | 0.96634 |
| Relative score ≥ 0.8 rule | 10.9 | 0.9703 | 0.96850 |
| Learned pruner q ≥ 0.001 | 9.3 | 0.9745 | 0.96976 |
| **Learned pruner q ≥ 0.003 (default)** | **7.5** | 0.9736 | **0.96966** |
| Learned pruner q ≥ 0.01 | 6.0 | 0.9709 | 0.96937 |

Tune it with `BER_PRUNE_Q` (`0` disables pruning).

---

## Repository layout

```
.
├── solution_v3/            Approach 1: Kaggle GPU pipeline (submitted)
│   ├── src/ber/            normalise, encoder, kNN blocking, features, matcher, decision
│   ├── src/kaggle/         the exact Kaggle jobs that produced the submission
│   ├── experiments/        full-scale analysis scripts
│   ├── submission_history/ v1, v2, v3 metrics
│   ├── METHODOLOGY.md      final write-up
│   └── EXPERIMENTS.md      every approach tried, with scores
├── src/ber/                Approaches 2 and 3: streaming CPU pipeline package
├── notebooks/              00_preprocessing … 05_inference_submission (run in order)
├── scripts/                run_notebooks.sh, validate_submission.py, EC2 helpers
├── CONTEXT.md              full project hand-over for the team
├── Documentation_template.md
└── requirements.txt
```

---

## Documentation

| Document | What's inside |
|---|---|
| [`solution_v3/METHODOLOGY.md`](solution_v3/METHODOLOGY.md) | Methodology write-up of the submitted solution |
| [`solution_v3/EXPERIMENTS.md`](solution_v3/EXPERIMENTS.md) | Every experiment, what it measured, and whether it was kept |
| [`solution_v3/submission_history/`](solution_v3/submission_history/) | Metrics of each submission (v1, v2, v3) |
| [`CONTEXT.md`](CONTEXT.md) | Data facts, all experiments on both pipelines, limits and next steps |

**What we learned across all approaches**

- Every true pair shares its `country`, and every S2/S3 record belongs to at most one S1 entity.
- ~43% of true pairs have different cleaned names but most share address tokens, so an address view is essential.
- More training data helps (~+0.001 to +0.002 F0.5 per doubling); bigger trees do not.
- The realistic ceiling is about **0.994**: blocking misses ~1% of true pairs, and thousands of address-less records carry a name shared by several S1 entities.

---

## Licences and rules

- No external data, APIs, geocoding or pretrained weights. Every model is trained from scratch and
  far below the 8B-parameter limit.
- All dependencies are open source (MIT / BSD / Apache-2.0 / ISC): polars, rapidfuzz, LightGBM,
  XGBoost, PyTorch, numpy, anyascii.
