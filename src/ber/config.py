"""Central configuration for the Business Entity Resolution pipeline.

Every notebook and the CLI read settings from here, so an instance change or a new
S3 bucket is a one-line edit (or an environment variable) rather than a hunt through
six notebooks.

Environment variables override the defaults below:
    BER_S3_BUCKET   S3 bucket name (no s3:// prefix). Unset/placeholder -> local only.
    BER_S3_PREFIX   Key prefix inside the bucket ("" = bucket root, the default).
    BER_WORK_DIR    Local working directory for raw-data cache + checkpoints.
    BER_RAW_DIR     Local folder that already holds dataset/{train,test}/*.tsv
                    (optional; skips the S3 download entirely when present).
    BER_N_JOBS      CPU workers (default: all cores).
    BER_SAMPLE_FRAC Fraction of S1 entities to keep for a fast dev run (e.g. 0.02).
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# S3 location. >>> FILL THESE IN (or export the env vars) <<<
# --------------------------------------------------------------------------------------
S3_BUCKET = os.environ.get("BER_S3_BUCKET", "amzn-ml-challenge-s3-team-gradient-descendants")
S3_PREFIX = os.environ.get("BER_S3_PREFIX", "").strip("/")  # "" = bucket root

# Raw files live at s3://{S3_BUCKET}/[{S3_PREFIX}/]dataset/{split}/{split}_source{1,2,3}.tsv
RAW_S3_SUBDIR = "dataset"


# Only files up to this size are mirrored to S3 (None = everything). Keeps usage inside the
# S3 free tier (5 GB, ~2k PUT/month); larger intermediates live on the EBS volume, which
# persists across instance stop/start.
S3_SYNC_MAX_MB = float(os.environ.get("BER_S3_SYNC_MAX_MB", 25)) or None


def s3_enabled() -> bool:
    """S3 sync is on only when a real bucket name has been configured."""
    return bool(S3_BUCKET) and S3_BUCKET not in {"YOUR-BUCKET-NAME", "none", "None"}


# --------------------------------------------------------------------------------------
# Local paths
# --------------------------------------------------------------------------------------

_raw_env = os.environ.get("BER_RAW_DIR")
# Optional pre-existing local copy of the dataset (e.g. the unzipped student_resource).
RAW_LOCAL_DIR = Path(_raw_env).resolve() if _raw_env else None

# --------------------------------------------------------------------------------------
# Compute sizing — tuned for a free-tier-class instance (2 vCPU / 8 GB RAM + swap)
# --------------------------------------------------------------------------------------
N_JOBS = int(os.environ.get("BER_N_JOBS", os.cpu_count() or 4))
CLEAN_CHUNK_ROWS = 250_000      # raw rows per streamed cleaning chunk
QUERY_CHUNK_ROWS = 100_000      # query rows per sparse top-K matmul batch
FEATURE_CHUNK_ROWS = 1_000_000  # candidate pairs per feature / scoring part file

# Fast dev mode: keep only this fraction of S1 entities (+ their matches and the
# same fraction of unmatched S2/S3 records). None = full data.
_sf = os.environ.get("BER_SAMPLE_FRAC")
SAMPLE_FRAC: float | None = float(_sf) if _sf else None
RANDOM_STATE = 42

# Checkpoints are namespaced by run mode so a dev-sample run can never be mistaken for
# (and skipped in place of) a full run:  pipeline_artifacts/  vs  pipeline_artifacts_sample0.02/
_RUN_TAG = f"_sample{SAMPLE_FRAC:g}" if SAMPLE_FRAC else ""
# Checkpoints are mirrored to s3://{S3_BUCKET}/[{S3_PREFIX}/]{CHECKPOINT_S3_SUBDIR}/...
CHECKPOINT_S3_SUBDIR = "pipeline_artifacts" + _RUN_TAG

REPO_DIR = Path(__file__).resolve().parents[2]  # .../business_entity_resolution
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", REPO_DIR / "work")).resolve()
RAW_CACHE_DIR = WORK_DIR / "raw"                     # local copy of the raw TSVs from S3
CHECKPOINT_DIR = WORK_DIR / ("checkpoints" + _RUN_TAG)  # parquet / model artifacts
OUTPUT_DIR = REPO_DIR / ("output" + _RUN_TAG)           # matching_results.tsv, candidate_pairs.tsv

# --------------------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------------------
# Retrieval runs from the S2/S3 side: every S2/S3 record belongs to at most one S1
# entity (verified in 01_eda), so "top-K S1s per S2/S3 record" needs a much smaller K
# for the same recall than "top-K S2/S3 per S1" (an S1 can have 11 matches).
TOPK = 15          # neighbours per S2/S3 record from the combined retrieval vector
# Weights of the parts of the combined retrieval vector (cos = sum_k w_k cos_k). The phonetic
# skeletons recover phonetically transliterated names/places; measured on the full India
# block: recall@10 96.5% (name+addr) -> 97.2% (4 equal parts), recall@30 97.3% -> 98.0%.
RETRIEVAL_WEIGHTS = {"name": 0.25, "phon": 0.25, "addr": 0.25, "addrp": 0.25}
MIN_SIM = 0.05     # drop neighbours below this cosine (pure noise)
TIE_EPS = 0.02     # rivals within this retrieval score of a record's best count as near-ties

# Candidate pruning (cascade stage 2b). A small LightGBM on retrieval-stage features only scores
# every retrieved pair; pairs with q < PRUNE_Q are dropped before the (expensive) full matcher.
# The survivors ARE candidate_pairs.tsv. Measured on the full India block: 70.4 -> 7.5 candidates
# per S1 entity at PRUNE_Q=0.003 for -0.0002 F0.5 (0.001: 9.3/S1, -0.0001; 0.01: 6.0/S1, -0.0005).
# Set BER_PRUNE_Q=0 to disable pruning.
PRUNE_Q = float(os.environ.get("BER_PRUNE_Q", 0.003))
PRUNER_TRAIN_PCT = 6.0   # % of S1 entities (taken from the train role) whose pairs train the pruner
# N-grams/tokens present in more than this many S1 records are dropped from the
# retrieval index. They carry almost no identity signal ("llc", " st", "private")
# and dominate sparse-matmul cost.
RETRIEVAL_MAX_DF = 20_000

# --------------------------------------------------------------------------------------
# Model — entity-level subsample (keeps the training table ~2 GB on an 8 GB instance)
# --------------------------------------------------------------------------------------
TRAIN_PCT = float(os.environ.get("BER_TRAIN_PCT", 30.0))  # % of S1 entities used for fitting
VALID_PCT = float(os.environ.get("BER_VALID_PCT", 4.0))   # % held out: half early-stop/threshold, half report
# Easy negatives (retrieval score < EASY_REL x the record's best) are 88% of training negatives
# but hold only 0.45% of positives; keeping a weighted EASY_NEG_KEEP sample of them cuts the
# training table ~4x (measured on the full India block: F0.5 -0.0009), which pays for training
# on ~4x more entities (+0.002 F0.5 per doubling). Validation roles always keep every pair.
EASY_REL = 0.8
EASY_NEG_KEEP = 0.10 if not PRUNE_Q else 1.0   # pruning already removes most easy negatives


def train_valid_pct() -> tuple[float, float]:
    """In dev-sample mode the sample is already small: use it all (70% train / 30% valid)."""
    return (70.0, 30.0) if SAMPLE_FRAC else (TRAIN_PCT, VALID_PCT)


ONE_TO_ONE = False       # optional: keep only the best S1 per S2/S3 record at inference
F_BETA = 0.5
TARGET_F05 = 0.985

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.08,
    num_leaves=127,
    min_child_samples=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    verbose=-1,
)
LGB_NUM_BOOST_ROUND = 3000
LGB_EARLY_STOPPING = 100


def ensure_dirs() -> None:
    for d in (RAW_CACHE_DIR, CHECKPOINT_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    return "\n".join(
        [
            f"S3 sync       : {'ON  s3://' + S3_BUCKET + '/' + S3_PREFIX if s3_enabled() else 'OFF (local only - set BER_S3_BUCKET)'}",
            f"WORK_DIR      : {WORK_DIR}",
            f"RAW_LOCAL_DIR : {RAW_LOCAL_DIR}",
            f"N_JOBS        : {N_JOBS}",
            f"SAMPLE_FRAC   : {SAMPLE_FRAC}",
            f"CHECKPOINTS   : {CHECKPOINT_DIR}  (S3: {CHECKPOINT_S3_SUBDIR}/)",
        ]
    )
