# Business Entity Resolution — Amazon ML Challenge 2026

Match every Source-1 business to all Source-2 / Source-3 records of the same real-world entity.
Metric: F0.5 per S1 entity, macro-averaged (singletons included).

The whole pipeline runs on the **full dataset on an AWS free-tier-class instance**
(m7i-flex.large: 2 vCPU, 8 GB RAM + 4 GB swap, 30 GB gp3) — every stage streams or chunks.

## Pipeline

```
raw TSVs (S3) ─► 00 clean ─► 01 EDA ─► 02 block + TF-IDF top-K ─► learned pruner ─► 03 features ─► 04 LightGBM + threshold ─► 05 test inference
```

| stage | method | checkpoint (local EBS; small artifacts also on S3) |
|---|---|---|
| 00 preprocessing | streamed; ASCII transliteration (anyascii), legal-form canonicalisation, alias (fka/dba) split, address normalisation (state removal, abbreviations, leading zeros, PO boxes), **phonetic consonant skeletons** of name and address | `clean/{split}_{s1,s2,s3}.parquet` |
| 02 candidates | hard block on `country`; inside a block, sparse TF-IDF top-15 from each S2/S3 record into S1 on one vector = name char-3grams ⊕ phonetic-name char-3grams ⊕ address word 1-2grams ⊕ phonetic-address word 1-2grams (equal weights). Only one block's S1 index in memory; competition statistics accumulated while streaming | `candidates/{split}/{country}.parquet`, `candidates/{split}_stats.npz` |
| 02b pruning | cascade stage: a small LightGBM on retrieval-stage signals only (score + name/address parts, rank, competition, near-ties, name/address frequency) drops retrieved pairs with q < `PRUNE_Q` (0.003). The survivors are the final candidate set = `candidate_pairs.tsv` = the only pairs the matcher scores. Full India block: 70.4 → 7.5 candidates per S1 for −0.0002 F0.5; ~8× less feature/scoring time | `pruned/{split}/{country}.parquet`, `model/pruner.txt` |
| 03 features | ~70 country-agnostic features: rapidfuzz name / phonetic / address similarities, TF-IDF cosines, numeric-token agreement, legal form, retrieval score split into name / address parts, **competition** (rank, relative score, margin to the best rival — per record and per S1), near-tie counts, **ambiguity** (how many S1 share this exact name / address) | `features/train/*.parquet` |
| 04 model | LightGBM on an entity-disjoint subsample: 30% of S1 entities train (all positives + hard negatives + a weighted 10% sample of easy negatives), 2% valid-A (early stopping + threshold), 2% valid-B (untouched report, every pair kept) | `model/lgbm.txt`, `model/meta.json` |
| 05 inference | same functions on test; ~150M pairs featurised and scored in a stream (only probabilities stored); threshold | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

Key data facts behind the design (verified in 01_eda): every labelled true pair shares its
`country`; every S2/S3 record belongs to at most one S1 entity; ~43% of true pairs have
different cleaned names but most share address tokens; many S2/S3 names are phonetic
transliterations of Indic-script names. The test set adds France (unseen in training), so no
component is keyed on country values.

## Layout

```
src/ber/          importable pipeline package (config, io, cleaning, blocking, features, metrics, model, pipeline)
notebooks/        00_preprocessing … 05_inference_submission (documented, run in order)
scripts/          run_notebooks.sh (headless run), validate_submission.py (official checker), connect_ec2.ps1
requirements.txt
```

## Reproduce

1. Raw data at `s3://<bucket>/dataset/{train,test}/…` (or point `BER_RAW_DIR` at a local folder
   containing `train/` and `test/`).
2. Environment:
   ```bash
   python3 -m venv ~/ber-venv && source ~/ber-venv/bin/activate
   pip install -r requirements.txt
   export BER_S3_BUCKET=<bucket>        # omit (or "none") to run without S3
   # export BER_SAMPLE_FRAC=0.02        # optional fast dev run on a consistent 2% sub-sample
   ```
3. Run all notebooks headlessly (or open them in JupyterLab, top to bottom):
   ```bash
   tmux new -d -s run "bash scripts/run_notebooks.sh > ~/run.log 2>&1"   # 00 → 05
   ```
4. Validate:
   ```bash
   python scripts/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir <folder with test_source1.tsv>
   ```

Every stage is checkpointed and resumes from the last finished part (set `FORCE = True` in a
notebook to recompute). Dev-sample runs use separate `checkpoints_sample<frac>/` and
`output_sample<frac>/` folders so they never collide with a full run.

## Compute and cost

- CPU only. On the free-tier instance: cleaning ~5 min / split, retrieval ~1.5 h / split,
  training features ~30 min, model ~15 min, streamed test scoring ~2–3 h.
- S3: raw data + model + submission only (large intermediates stay on the EBS volume, which
  survives stop/start) — inside the S3 free tier (5 GB, ~2k PUT/month). S3→EC2 transfer in the
  same region is free.

## Licences

All dependencies are open source (MIT / BSD / Apache-2.0 / ISC). The matching model is a
LightGBM gradient-boosted tree ensemble trained from scratch (no pretrained weights).
