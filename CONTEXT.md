# Project context — Amazon ML Challenge 2026 · Business Entity Resolution

Hand-over document for the team (GradientDescendants). It summarises everything decided,
built, measured and learned so far, so anyone can continue in their own workspace.

- Repo: https://github.com/AvinashOraon123/Amazon_ML_Challenge_team_GradientDescendants (private)
- **Latest (2026-09-27): best validation F0.5 0.9880, leaderboard 0.9821, from `solution_v3/`; see section 0.**
- Status at time of writing (2026-09-26 ~08:00 UTC):
  - Full-data train + validation **done**. Held-out macro F0.5 = **0.9740** (target 0.985).
  - Test inference (notebook 05) **running** on Avinash's free-tier EC2, ETA ~16:00–16:30 UTC.
    It writes a validated `output/matching_results.tsv` + `output/candidate_pairs.tsv`.

---

## 0. Update 2026-09-27: new best (validation 0.9880, leaderboard 0.9821) from `solution_v3/`

> Read this section first. Sections 1-10 below describe the notebook pipeline as of 2026-09-26
> (validation 0.9740) and are kept unchanged as history; the data facts in section 2 still hold.

### Where we are

| | Notebook pipeline (sections 3-10) | **Kaggle GPU pipeline `solution_v3/` (submitted)** |
|---|---|---|
| Held-out macro F0.5 | 0.9740 (valid-B) | **0.9880** (1.1M held-out S1 entities; 0.9880 on a fixed 10% subset too) |
| Public leaderboard | – | **0.9821** (0.982127) |
| Candidates per S1 (test) | 70.4 (7.5 with the pruner branch) | **8.7** |
| True pairs kept by candidates | 97.5% (India) | **99.0%** |
| Singleton F0.5 | 0.9655 (accuracy) | 0.9852 |

- The submitted files are the v3 outputs: `matching_results.tsv` (98,717 empty rows, 5.85M matches) and
  `candidate_pairs.tsv` (15.1M pairs). Both passed `validate_submission.py --check-ids`.
- The final zip `Gradient_Descendants_submission.zip` (output/, code/business_entity_resolution/,
  filled Documentation_template.md) was built from `solution_v3/`; it is kept outside the repo because
  of its size (128 MB).
- Docs: `solution_v3/README.md` (how to run), `solution_v3/METHODOLOGY.md` (final write-up),
  `solution_v3/EXPERIMENTS.md` (every approach tried, with scores), `solution_v3/submission_history/`
  (v1, v2, v3 metrics; v2 methodology).

### How `solution_v3` differs from the notebook pipeline

| Stage | Notebook pipeline | `solution_v3` |
|---|---|---|
| Cleaning | `cleaning.py` rules + phonetic skeletons | similar rules (`normalize.py`) + cleaned copies for the matcher (filler words, phone numbers, PO boxes, URLs, "ICTY/CDP" junk removed), legal-form sets, 12 record-level **noise markers** |
| Blocking | sparse TF-IDF top-15 per S2/S3 record (+ learned pruner branch) | **hashed character n-gram bi-encoder** trained from scratch (contrastive, mined hard negatives) with joint / address / name embeddings; exact GPU top-k per country; policy: joint top-5, address top-2 (cos > 0.8), name top-5 only for address-less records, cap 15 candidates per S1 per view |
| Train / validation split | 30% train entities, valid-A 2%, valid-B 2% | **50% of S1 entities held out**: the encoder trains on the other half, the matcher trains 5-fold on candidates of the held-out half (entities the encoder never saw) |
| Matcher | LightGBM, ~67 features | **XGBoost on GPU** (LightGBM on CPU gives the same score), ~150 features incl. name frequency, competition vs the record's other candidates, token accounting, house-number distance, noise markers; stage-2 stacker |
| Decision | one threshold | each record keeps its best S1, then per-S1 expected-F0.5 prefix (empty when P(no match) wins) |
| Compute | free-tier EC2 CPU | free Kaggle T4 via the Kaggle API; about 3.5 h end to end, resumable |

### How the score got from 0.9856 to 0.9880 (details in `solution_v3/EXPERIMENTS.md`)

1. **v1 (0.9856)**: bi-encoder + multi-view blocking + LightGBM. Error analysis: 24,840 missed pairs vs
   3,436 wrong merges; 61% of misses were address-less records.
2. **v2 (0.9860)**: name-frequency features (an address-less record whose exact name belongs to one S1
   is that S1's in 97.7% of cases), competition features, name-view search for address-less records;
   candidates 10.6 -> 8.1 per S1.
3. **Full-scale local experiments** on the v2 feature table: cleaned text + legal form + noise markers
   **+0.0016**; learning curve **+0.0009 per doubling** of matcher training rows.
4. **v3 (0.9880)**: those features + the 50% entity holdout (5x more matcher data) + XGBoost on GPU.

Confirms the notebook team's finding in section 5: more data helps (~+0.001 to +0.002 per doubling),
bigger trees do not.

### What did not help (full scale, exact metric)

| Idea | Effect |
|---|---|
| Bigger / depth-wise trees, averages of up to 4 GBDTs | at most +0.00015 |
| Entity-level "has any match" gate for singletons | +0.00001 |
| Specialist model for S1 entities with one strong candidate | +0.00014 |
| Assignment features from an entity's other records per source | +0.0002 |
| MLP blended with the GBDT | never won the blend |

### Limits we measured

- **Realistic ceiling about 0.994**: a perfect matcher still loses the 0.95% of true pairs blocking misses
  and ~7.2K address-less records whose name is shared by several S1 (e.g. six "Laex Inc").
- **Singletons**: 89.5% of singleton false matches are decoy records that are near-exact noisy copies of
  the singleton (same address, typo-level name change); the text cannot separate them.
- **Leaderboard gap (0.9880 -> 0.9821)**: not measurable without test labels; the likely causes are
  France (15% of test, generic repeated names like "Ecole", "Lycée", "Amicale"; about 0.95 if US/India
  hold their validation level) and a denser test set (5.75 S2/S3 records per S1 vs 4.7 in train).

### Infrastructure added

- Kaggle account `kshirodkalet`: private datasets `ber-challenge-data` (lossless parquet copy of the
  TSVs) and `ber-code` (the `solution_v3/src` package); GPU jobs `ber-train-encoder`, `ber-final`,
  `ber-final-resume` (scripts in `solution_v3/src/kaggle/`).
- Kaggle discards all output of a job that ends in an error, so the job scripts never raise and write
  `PIPELINE_STATUS.txt`; a resume job reuses the saved encoder, candidate rule and searches.
- Gotchas hit: GPU memory held by PyTorch's cache starves XGBoost (clear it first); a 9.5M x 130 feature
  matrix needs record-aligned chunking on 29 GB RAM; the job script must be compile-checked before
  pushing.

### Suggested next steps

1. Validate with extra decoys so each S1 faces ~5.75 records (test density) and re-tune the decision rule.
2. Treat generic French institution words (école, lycée, association, comité, centre, amicale, société)
   as low-weight filler for France.
3. Save test probabilities so decision changes can be tried in minutes instead of rerunning the test side.
4. Try combining the notebook team's record-level ranker / pruner ideas with the `solution_v3` features.

---

## 1. The problem (short)

Three sources of business records: **S1** (deduplicated reference), **S2**, **S3**. For every S1
entity, find all S2/S3 records describing the same real-world business (0, 1 or many).

- Files (TSV, tab-separated, no quoting): `entity_id, business_name, business_address, country`.
  Ground truth: `source1_entity_id, matched_entity_ids` (comma list, empty = singleton).
- Train: S1 2.21M, S2 5.03M, S3 5.29M rows. Test: S1 1.73M, S2 4.89M, S3 5.08M.
- Countries: train = US + India; **test adds France** (never seen in training) → nothing may be
  keyed on country values.
- **Metric: F0.5 per S1 entity, macro-averaged over ALL S1 entities** (singletons included:
  empty prediction for a singleton = 1.0, any prediction = 0.0). Precision weighted 2×.
- Submission: `matching_results.tsv` (scored) + `candidate_pairs.tsv` (the exact pairs the model
  scored; every matched id must be a candidate). One row per test S1, unique ids, TAB, UTF-8.
  Checker: `scripts/validate_submission.py` (official, copied from the kit).
- Rules: no external data / APIs / geocoding. Final model must be MIT/Apache and ≤ 8B params.
  Final zip = `output/` + `code/business_entity_resolution/` (src, README, requirements) +
  filled `Documentation_template.md`.

## 2. Key data facts (verified on full train data)

| Fact | Consequence |
|---|---|
| 100% of true pairs share `country` | country = lossless hard block (France = its own block) |
| every S2/S3 record belongs to **at most one** S1 | retrieve from the S2/S3 side; "competition" features; optional one-to-one assignment |
| 5.58% of S1 are singletons; S1 averages ~3.5 matches (max 11) | singleton false merges cost a full point each |
| only 56.8% of true pairs share the exact cleaned name | the address must drive retrieval as much as the name |
| 81.4% of true pairs share ≥1 address number | numeric-token features are strong |
| ~4–5% of S2/S3 names are Devanagari/Kannada/Telugu; after transliteration they are *phonetic* ("brait prodyusr praivet" = "Bright Producer Private") | phonetic skeleton keys |
| noise: legal-suffix variants, `fka/dba` aliases, web-domain names, `(ID: 123)`, digit-for-letter typos (`N0rth`), repeated tokens, random replacement names ("Irijaxnex"), zero-padded / mutated house numbers, `Door No`/`#`/`H.No` prefixes, state names in native script, PO boxes | cleaning rules in `src/ber/cleaning.py` |
| **the data deliberately contains near-copy NON-matches** (same name, house number changed) and true matches with mutated numbers / truncated addresses | a hard, partly ambiguous decision boundary |
| many S1 share generic names (a no-address record's name is shared by ~18 S1 on average among the hard misses) | "ambiguity" (name/address frequency) features |

## 3. Pipeline (what the code does)

```
00 clean ─► 01 EDA ─► 02 block + TF-IDF top-K ─► 03 features ─► 04 LightGBM + threshold ─► 05 test inference
```

Everything is in the importable package `src/ber/`; notebooks call the same functions for train
and test (identical preprocessing). Built to run the **full data on a free-tier instance**
(2 vCPU, 8 GB RAM + 4 GB swap, 30 GB disk): every stage streams / chunks and is checkpointed
(resumable). Dev mode: `BER_SAMPLE_FRAC=0.02` = consistent 2% sub-world (separate checkpoint
folders `checkpoints_sample0.02/`, `output_sample0.02/`, never collides with full runs).

| Stage | Method | Checkpoint |
|---|---|---|
| 00 | streamed cleaning (anyascii transliteration, legal-form canonicalisation, alias split, address normalisation, state removal, abbreviations, leading zeros, PO boxes) + **phonetic consonant skeletons** (`name_phon`, `addr_phon`) | `clean/{split}_{s1,s2,s3}.parquet`, `clean/train_gt.parquet` |
| 02 | country block; inside a block sparse TF-IDF top-**15** from each S2/S3 record into S1 on one vector = name char-3grams ⊕ phonetic-name char-3grams ⊕ address word 1-2grams ⊕ phonetic-address word 1-2grams (equal weights, `max_df` 20k). Only one block's S1 index in memory; competition stats accumulated while streaming; per-block resume | `candidates/{split}/{country}.parquet`, `candidates/{split}_stats.npz` |
| 03 | ~67 country-agnostic features per pair (rapidfuzz name/phonetic/address similarities, TF-IDF cosines, numeric-token agreement, legal form, retrieval score + name/address parts + rank, **competition** per record & per S1: rel score, margin to best rival, best-vs-2nd gap, near-ties; **ambiguity**: # of S1 sharing exact name/address). Train entities: all positives + hard negatives + weighted 10% of easy negatives | `features/train/{country}-{k}.parquet` |
| 04 | LightGBM on entity-disjoint roles: **train 30%** of S1 entities, **valid-A 2%** (early stopping + threshold), **valid-B 2%** (untouched report). Threshold maximises the leaderboard macro F0.5 | `model/lgbm.txt`, `model/meta.json` |
| 05 | same functions on test; ~150M pairs featurised + scored in a stream (only probabilities stored); threshold; writes + validates submission | `scores/test/*.parquet`, `output/*.tsv` |

### File map
```
src/ber/config.py     all knobs (env-overridable): S3, paths, chunk sizes, TOPK, weights, TRAIN_PCT, LGB params
src/ber/io.py         S3 (boto3) + local checkpoints; only files <= 25 MB mirrored to S3 (free tier)
src/ber/cleaning.py   name/address normalisation + phonetic skeletons
src/ber/blocking.py   hashing TF-IDF, multi-part combined vectors, sparse top-K (sparse_dot_topn)
src/ber/features.py   pair features (vectorised; per chunk)
src/ber/pipeline.py   stage functions: run_clean / run_candidates / run_features / run_scoring / write_submission,
                      entity_roles, load_training_matrices, progress.json
src/ber/model.py      LightGBM train/predict helpers
src/ber/metrics.py    exact leaderboard metric (sweep_idx, per_entity_f05), expected-F rule
notebooks/00..05      documented pipeline (run in order)
scripts/run_notebooks.sh   headless runner (tmux)       scripts/dashboard.ps1 + ber_status.sh   live dashboard
scripts/connect_ec2.ps1    SSH + Jupyter tunnel helper   scripts/validate_submission.py         official checker
```

### Important config knobs (`src/ber/config.py`, env vars in brackets)
`TOPK=15`, `RETRIEVAL_WEIGHTS` (4 equal parts), `RETRIEVAL_MAX_DF=20000`, `TIE_EPS=0.02`,
`TRAIN_PCT=30 [BER_TRAIN_PCT]`, `VALID_PCT=4 [BER_VALID_PCT]`, `EASY_REL=0.8`, `EASY_NEG_KEEP=0.1`,
`LGB_PARAMS` (127 leaves, lr 0.08, early stop 100), `N_JOBS [BER_N_JOBS]`, `SAMPLE_FRAC [BER_SAMPLE_FRAC]`,
`S3_BUCKET [BER_S3_BUCKET]` (default = team bucket), `S3_PREFIX [BER_S3_PREFIX]` (default "" = bucket root),
`BER_WORK_DIR`, `BER_RAW_DIR` (local dataset folder with `train/` and `test/` — skips S3).

**Validation roles are a pure hash of the S1 entity id** (`pipeline.entity_roles`): every machine
gets the same valid-A / valid-B entities, and the validation ranges come first so changing
`TRAIN_PCT` never moves them. **Always report valid-B macro F0.5** — it is the team scoreboard.

## 4. Results history

| Version | Data | valid-B macro F0.5 |
|---|---|---|
| first model (3 retrieval channels, K=5 each) | 2% sample | 0.9918 (OOF) |
| lean free-tier rewrite (single channel K=10) | 2% sample | 0.9890 |
| + fixed competition-feature indexing bug + ambiguity features | 2% sample | 0.9918 |
| + phonetic parts in retrieval + phonetic features, K=15 | 2% sample | 0.9917 |
| same model, **India block only, full scale**, 8% train | full | **0.9698** (P 0.990, R 0.937, cand. ceiling 0.975) |
| **full data (US+India), 30% train entities, easy-neg subsampling** | full | **0.9740** (P 0.9934, R 0.9436, singleton acc 0.9655, cand. ceiling 0.9817, thr 0.72) |

⚠️ The 2% sample is optimistic (50× fewer look-alike distractors). Only trust full-scale numbers.

## 5. Experiments and what they taught us (full-scale unless noted)

**Retrieval (full India block, 10–20k real queries, recall of the true S1 in top-K):**
| Setting | recall@10 | recall@30 | cost (s / 100k queries, 2 threads) |
|---|---|---|---|
| name+addr, max_df 20k | 96.5% | 97.3% | ~78–102 |
| max_df 5000 / 2000 / 1000 / 500 | 93.3 / 91.3 / 89.9 / 88.3% | — | 15 / 6 / 4 / 3 |
| + phonetic name (0.35/0.15/0.5) | 97.1% | 98.0% | ~197 |
| **4 equal parts (current)** | **97.2%** | 98.0% | ~131–164 |
| 4 parts but capped phonetic df | 96.2–96.5% | — | 76–128 |
| two-stage (pruned shortlist @200 then rerank) | shortlist ceiling only 95.8–96.9% @200 | | 29–42 |
- Query chunk size barely matters (2k: 180 s, 10k: 128 s, 40k: 119 s per 100k).
- Recall@200 is still only ~98.5% → remaining misses are genuinely hard (random names + truncated
  addresses, empty-address same-name ties).

**Model / decision (India full-scale validation):**
| Experiment | valid-B F0.5 |
|---|---|
| baseline LightGBM (127 leaves) | 0.9698 |
| half the training data | 0.9675 (→ ~+0.002 per doubling of data) |
| 255 leaves / 511 leaves lr 0.05 | 0.9687 / 0.9688 (no gain: information-limited, not capacity-limited) |
| second-stage model with S1-context features (p1 of other candidates, similarity to best co-candidate) | 0.9683 |
| per-entity expected-F0.5 selection instead of one threshold | 0.9697 |
| keep 10% / 5% of easy negatives (weighted) | 0.9689 / 0.9687 (cheap: 4× fewer rows) |
| one-to-one assignment (sample only) | ~+0.00004, negligible |

**Where the loss is (India):** FN inside candidates 51% of lost F; entities with matches but nothing
predicted 19%; extra FPs 17%; singleton false merges 7.5%; 2.5% of true pairs never reach the
candidates. Among FNs: 27% have an **empty S2/S3 address** (vs 2% of TPs) and names shared by
many S1s; 37% have the true S1 at retrieval rank ≥ 2.

## 6. Timings on the free-tier instance (m7i-flex.large, full data)
- Cleaning: ~5–9 min / split (incl. profiling). EDA: ~30 s.
- Candidates: India 2h24m, US ~3h40m → **~6.1 h per split** (the bottleneck; ~180 s / 100k queries).
- Train features (30% entities, subsampled): ~15 min. Training: ~40 min (1,956 trees). 04 total ~54 min.
- Test scoring: ~2.5 h estimated (~150M pairs). One full iteration ≈ 15 h.
- Peak RAM: candidates ~5 GB, training ~4.5 GB (after fixes).

## 7. Infrastructure (AWS free tier only — team decision: no paid services)
- Region **ap-south-1 (Mumbai)**; bucket `amzn-ml-challenge-s3-team-gradient-descendants`, raw data at
  `s3://<bucket>/dataset/{train,test}/`. Checkpoints mirror to `pipeline_artifacts/` (only ≤ 25 MB files,
  to stay inside S3 free tier 5 GB / ~2k PUT per month).
- EC2: Ubuntu 26.04, **m7i-flex.large** (free-tier eligible, 2 vCPU / 8 GB), 30 GB gp3, 4 GB swap,
  IAM role `ber-ec2-role` with S3 access to the bucket only (no keys on the instance), SSH from My IP only.
- Python 3.14 venv `~/ber-venv`; settings in `~/ber.env` (read by `run_notebooks.sh`):
  `BER_S3_BUCKET=...`, `BER_S3_PREFIX=`, `BER_WORK_DIR=/home/ubuntu/ber_work`.
- Run: `tmux new -d -s run "bash ~/business_entity_resolution/scripts/run_notebooks.sh 00 01 02 03 04 05 > ~/run.log 2>&1"`
- Jupyter: `jupyter lab --no-browser --port 8888 --ip 127.0.0.1` in tmux + SSH tunnel
  (`scripts/connect_ec2.ps1`). Dashboard: `scripts/dashboard.ps1` (edit `-Ec2Dns`).
- Each teammate should use **their own** AWS account, bucket, key pair and role — never share keys.
  Update `S3_BUCKET` via `BER_S3_BUCKET` and the DNS/key path in the PowerShell helpers.

## 8. Gotchas we hit (save yourself the time)
1. **Your home folder may be a git repo** (Avinash's `C:\Users\AVINASH` is): run git only inside this project folder.
2. pandas 3 uses pyarrow strings → regex is RE2: write `"[\u0900-\u097F]"` (real chars), not raw `r"\u0900"`.
3. Raw TSVs need `quoting=3` (QUOTE_NONE) and `keep_default_na=False`.
4. Cleaned `country` is **lower-case** (`india`, `us`, `france`).
5. `pkill -f nbconvert` inside an SSH command kills the SSH session itself — use `pkill -f "[n]bconvert"`.
6. A long-lived tmux server keeps old env vars (it once re-enabled sample mode) → `run_notebooks.sh` unsets `BER_*` and reads `~/ber.env`.
7. Non-interactive SSH doesn't read `~/.bashrc` → use `~/ber.env`.
8. Loading all feature parts into one DataFrame OOM-kills an 8 GB box → use `P.sample_features` / `P.load_training_matrices(roles=...)`.
9. Windows PowerShell mangles quotes in `ssh host "cmd"` → `connect_ec2.ps1` sends the remote script base64-encoded.
10. Windows laptops: joblib/loky prints harmless `resource_tracker KeyError` tracebacks — not errors.
11. Local laptop with ~8 GB is slower than the EC2 instance for retrieval (swapping); Claude Code's background jobs get auto-killed under memory pressure unless started with `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`.
12. Competition stats are **global-index** arrays — index them with `s1_idx` / `r_idx`, not block-local positions (this bug cost ~0.003 early on).

## 8b. Organiser update — smaller candidate sets rank higher (NEW)
The organisers announced that `candidate_pairs.tsv` and its code are reviewed for the final ranking
and that **a smaller candidate set per S1 entity ranks higher**. `candidate_pairs.tsv` must be the
exact set the model runs inference on (the last filtering stage).

Our answer (branch `candidate-pruning`, not yet on the EC2 run that produced 0.9740):
**cascade with a learned pruner** — retrieval top-15 → small LightGBM on retrieval-stage signals
only (`features.retrieval_features`) → keep q ≥ `PRUNE_Q` → full matcher on the survivors.

Full-scale India measurement (candidates per S1 / candidate recall / final valid-B F0.5):
| stage | cand/S1 | recall | F0.5 |
|---|---|---|---|
| retrieval top-15 (current submission) | 70.4 | 0.9753 | 0.96983 |
| rank ≤ 2 rule | 9.1 | 0.9547 | 0.96634 |
| rel ≥ 0.8 rule | 10.9 | 0.9703 | 0.96850 |
| **learned pruner q ≥ 0.001** | **9.3** | 0.9745 | **0.96976** |
| **learned pruner q ≥ 0.003 (default)** | **7.5** | 0.9736 | **0.96966** |
| learned pruner q ≥ 0.01 | 6.0 | 0.9709 | 0.96937 |
Side effect: features/scoring only run on survivors → test scoring ~20 min instead of ~2.5 h.
Code: `pipeline.train_pruner`, `pipeline.run_prune`, `candidate_files(split, pruned=...)`;
notebooks 02 (section 4) and 05 (section 2b). Tune with `BER_PRUNE_Q` (0 disables).

## 9. Team plan (4 people, free tier; 16 GB laptop available)
Shared rules: branch per person, PR only if **valid-B > 0.9740**, same `entity_roles` everywhere,
exchange only small per-pair outputs keyed by `(r_idx, s1_idx)`, final zip must reproduce from `src/`.

1. **Person 1 (Avinash, EC2):** finish test submission (safe fallback ≈ 0.974); then K=25 + larger
   `TRAIN_PCT`; integrate others' wins.
2. **Person 2 (16 GB laptop):** record-level **LightGBM ranker** (lambdarank grouped by S2/S3 record,
   rows sampled *by record* with all 15 candidates) + "no owner" margin; **noise-aware features**
   (gibberish-name score from a char n-gram LM trained on S1 names, noise-pattern flags,
   asymmetric number features). Most promising new signal.
3. **Person 3 (own EC2):** split candidate generation across accounts by country block (halves the
   6 h step); try **multi-key blocking** (exact name, phonetic name + city token, house number +
   street token, postcode) and compare with the 98.2% ceiling.
4. **Person 4 (Kaggle/Colab free GPU):** multilingual embeddings (e.g. `intfloat/multilingual-e5-small`
   or `paraphrase-multilingual-MiniLM-L12-v2`, MIT/Apache) + FAISS → recall comparison and an
   `emb_cos` / `emb_rank` feature per pair.
5. **Ensembling:** XGBoost/CatBoost alone ≈ LightGBM (±0.001); averaging diverse GBDTs ≈ +0.001–0.003;
   the bigger gains should come from blending *different* model families (ranker, embeddings).

Realistic outlook: GBDT tweaks + K=25 + more data ≈ 0.978–0.980. Reaching 0.985 needs new signal
(ranker / embeddings / noise features); part of the loss is ambiguity built into the data.

## 10. How to get started (teammate)
```bash
git clone https://github.com/AvinashOraon123/Amazon_ML_Challenge_team_GradientDescendants.git
cd Amazon_ML_Challenge_team_GradientDescendants
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# point at your data: either local folder with train/ and test/ ...
export BER_RAW_DIR=/path/to/student_resource/dataset
export BER_S3_BUCKET=none            # ...or your own bucket name
export BER_SAMPLE_FRAC=0.02          # fast 2% dev run first (~10 min)
bash scripts/run_notebooks.sh 00 01 02 03 04
```
Then unset `BER_SAMPLE_FRAC` for full runs (needs ~6 h/split for candidates on 2 vCPU; faster on
more cores via `BER_N_JOBS`). To experiment on features/models only, get a copy of the full
`candidates/` + `features/` checkpoints from Person 1 (via your own S3 bucket in ap-south-1) and
start from notebook 04.
