# Business Entity Resolution — reproducible pipeline

> **In this repository:** this is the submitted v3 pipeline (validation F0.5 0.9880, public leaderboard
> 0.9821), kept self-contained in `solution_v3/` so it does not touch the notebook pipeline at the repo
> root. Run the commands below from `solution_v3/src/` and point `--data` at your copy of the challenge
> `dataset/` folder (the data itself is not in the repository). The methodology write-up is
> `solution_v3/METHODOLOGY.md`; `experiments/` holds the full-scale analysis scripts, and
> `solution_v3/EXPERIMENTS.md` lists every approach tried and how the score reached 0.9880.

Everything runs from one command. Stages save their outputs and are skipped when re-run, so an
interrupted run resumes where it stopped (the encoder also resumes from its last finished epoch).

## Setup

```bash
pip install -r requirements.txt
```

No external data, APIs or pretrained models are used. Every model (character n-gram bi-encoder,
LightGBM, MLP) is trained from scratch on the provided training data, far below 8B parameters.
Libraries are permissively licensed (polars MIT, rapidfuzz MIT, LightGBM MIT, anyascii ISC,
torch BSD, numpy BSD).

## Run end to end

From `src/`, with the challenge files in `student_resource/dataset/{train,test}/*.tsv`:

```bash
cd src
# settings of the submitted run (v3): half of the Source-1 entities held out for the matcher, no MLP
export BER_HOLDOUT_PCT=50 BER_MLP=0 BER_GB_VARIANTS=lg127
python pipeline.py run-all \
    --data ../../../dataset \
    --work ../work \
    --models ../models \
    --out ../../../output
```

The matcher uses XGBoost on the GPU when one is available and LightGBM on the CPU otherwise
(`BER_GBDT=xgb|lgb` forces a backend). Without the environment variables the defaults are a 10% holdout
and a GBDT + MLP blend, which is what the earlier v1/v2 runs used.

This writes `output/matching_results.tsv`, `output/candidate_pairs.tsv` and `output/metrics.json`
(validation F0.5, blocking statistics, per-country diagnostics). Then validate:

```bash
cd ../../..        # student_resource/
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

A GPU is strongly recommended (the full v3 run took about 3.5 hours on one Kaggle T4 with 29 GB RAM;
features are built in record-aligned chunks so memory stays bounded). Every stage writes its outputs and
is skipped on a re-run, so an interrupted run resumes. `src/kaggle/kernel_final` and `kernel_resume` are
the exact Kaggle jobs that produced the submission (the second resumed the first from its saved encoder,
candidate rule and training searches).
`--data` may also point to a flat folder of lossless parquet copies of the TSVs
(`train_source1.parquet`, ...), which is how the Kaggle runs read the data.

Quick smoke test on 1% of the data (CPU, ~10 minutes):

```bash
python pipeline.py run-all --data ../../../dataset --work ../work_s1 --models ../models_s1 --out ../out_s1 \
    --sample 1 --epochs 2 --mine-from 1 --batch-size 512 --buckets-log2 18
```

## Stages (src/ber/)

| Stage | Module | Output |
|---|---|---|
| 1. Load + normalise text, fixed-width byte matrices | `data.py`, `normalize.py`, `prepare.py` | `work/<split>/records.parquet`, `bytes_*.npy` |
| 2. Train the hashed character n-gram bi-encoder (contrastive, mined hard negatives) | `encoder.py`, `train_encoder.py` | `models/encoder.pt`, `encoder_log.json` |
| 3. Candidate generation: multi-view kNN (joint / address / name embeddings), policy chosen on validation | `knn.py`, `blocking.py` | `work/<split>/neighbours.npz`, `models/policy.json` |
| 4. Pair features (embedding, fuzzy string, cleaned text, legal form, numbers, noise markers, name frequency, competition) | `features.py` | `work/<split>/features/*.parquet` |
| 5. Matcher: XGBoost (GPU) or LightGBM, 5-fold CV on held-out entities, stage-2 stacker | `matcher.py` | `models/matcher/` |
| 6. Decision: one entity per record, tuned threshold / expected-F0.5 rule | `decide.py` | `models/matcher/decision.json` |
| 7. Test inference and TSV writing | `run.py` | `output/*.tsv`, `output/metrics.json` |

Validation protocol: a share of Source-1 entities (50% in the submitted run via `BER_HOLDOUT_PCT`,
10% by default; chosen by a hash of the entity id) is held out.
The encoder never trains on them; the matcher trains and tunes only on candidates of these held-out
entities with out-of-fold predictions, and the reported F0.5 is the challenge metric over them.

Individual stages are also available: `pipeline.py prepare | train-encoder | block` (see `--help`).
