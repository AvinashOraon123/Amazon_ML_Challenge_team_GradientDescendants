# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Gradient_Descendants
**Team Members:** Avinash Oraon, Kshirod Kalet, Samprit Halder, Ayush Chanderia
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We treat entity resolution as *"which Source-1 entity does each Source-2/3 record belong to, if any?"*.
A hashed character n-gram **bi-encoder** trained from scratch (contrastive loss with mined hard
negatives) produces joint, address-only and name-only embeddings. **Multi-view GPU nearest-neighbour
search** with a per-entity hub cap yields **6.6 candidates per Source-1 entity** (8.7 on the test set)
while keeping **99.0% of true pairs**. A **gradient-boosted pair classifier (XGBoost on GPU)** over ~150
embedding, fuzzy-string, cleaned-text, number, legal-form, noise-marker, name-frequency and competition
features, a **stage-2 stacker** over competing candidates, a **one-entity-per-record** constraint and an
**expected-F0.5 decision rule** produce the final matches. The encoder is trained on half of the Source-1
entities so that the matcher can learn from the other half, which the encoder never saw.
Held-out validation macro F0.5: **0.9880** (1.1M held-out entities).

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from exploring the 2.2M / 5.0M / 5.3M training records and 7.64M labelled pairs:

- **No true pair crosses countries**, so every search runs within one country (an open set of labels;
  France in the test set is handled by the same code path with no country-specific branches).
- **Each Source-2/3 record matches at most one Source-1 entity** (verified on all 7.64M pairs). This turns
  the problem into a per-record assignment and is exploited both in blocking and in the final decision.
- 5.6% of Source-1 entities have no match (singletons); matched entities have 1-11 matches (mode 3);
  92-93% of matched entities have records in both Source 2 and Source 3.
- 26% of Source-2/3 records (2.68M) match nothing: they act as **distractors**, many of them deliberate
  **siblings** of a real entity: same street with a nearby house number (544 vs 557 Portofino Loop,
  8-2-603/M/19 vs /M/24) and one name word swapped ("Wright Horizon Equipment" vs "... Petroleum").
  True copies carry the same kinds of perturbation (6253 vs 6255 Highway 284), which bounds precision.
- Name noise: word shuffles and duplicated words ("Suryanth Suryanth Saors"), legal suffix moves/drops,
  OCR digits ("6eneral", "P0rtfolio"), spurious accents, domains ("indriyaclub.com"), phone numbers,
  acronyms ("SHLA" for "Smith, Holman and Love Alpex"), unrelated brand names at the true address
  ("Onyxxylo"), and 11-15% of names in Indic scripts (Devanagari, Gujarati, Bengali, Tamil, Kannada...).
- Address noise: case, abbreviations (RD/Road, R./Rue), zero padding ("0701"), dropped leading digits
  ("476" for 7476), state name vs code, native-script state names, "null" tokens, PO boxes / PMBs, URLs,
  appended place junk ("Darby CITY", "Baltimore ICTY", "Louisville CDP"), 3.3% empty addresses.
- **Address-less records**: 97.7% of them have a true match; when their core name belongs to exactly one
  Source-1 entity of the country, that entity is the match in 97.7% of cases, but when several entities
  share the name (e.g. six "Laex Inc") the record is genuinely ambiguous.
- **Noise markers are informative priors**: records with stray accents are matched 83% of the time,
  with OCR digits 82%, with a leading junk character 83%, domain-style names 96% (74% overall).

### 2.2 Solution Strategy

**Approach Type:** Hybrid: learned blocking (bi-encoder + multi-view kNN) + pairwise GBDT classifier +
stacked assignment/decision layer.
**Core Innovation:** a GPU-hashed character n-gram bi-encoder with three output views searched with an
exact grouped top-k; an entity-level train/validation split in which the encoder never sees the entities
the matcher learns from; and a record-to-entity assignment constraint with an expected-F0.5 decision rule.

Pipeline (every stage resumable; code in `code/business_entity_resolution/src/ber/`):

1. **Normalisation** (`normalize.py`, polars, rule-based, country-agnostic): transliteration with
   `anyascii`, lowercasing, accent/punctuation removal, domain unwrapping, legal-form stripping to a *core
   name*, a **consonant skeleton** of the core name (bridges transliteration: "constructions" ->
   `knstrktns`, Gujarati -> `knstrksns`), address abbreviation expansion (incl. French), zero-padding
   removal, state name -> code mapping. The matcher additionally uses *cleaned* copies: filler words,
   phone numbers, PO boxes, URLs and place junk removed; legal forms normalised (Inc/Incorporated, Pvt/
   Private...); record-level noise markers.
2. **Bi-encoder** (`encoder.py`, `train_encoder.py`).
3. **Candidate generation** (`knn.py`, `blocking.py`).
4. **Pair features** (`features.py`), built in record-aligned chunks written to parquet (bounded memory).
5. **Matcher** (`matcher.py`): XGBoost (GPU) or LightGBM (CPU), 5-fold, plus a stage-2 stacker.
6. **Decision** (`decide.py`).

**Validation protocol.** 50% of Source-1 entities (chosen by a hash of the entity id) are held out: the
encoder trains only on the other half; the matcher trains and tunes only on candidates of held-out
entities with 5-fold out-of-fold predictions (folds split by record so each record's competing candidates
stay together). All reported F0.5 values are the challenge metric over held-out entities, singletons
included. `validation_f05_orig10` reports the same metric on the fixed 10% subset used by earlier runs.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** no hand-made keys; learned embeddings.
  - Bi-encoder: each record's normalised name, skeleton and address are fixed-width byte arrays; character
    2-5-grams are **hashed on the GPU** into 4M buckets, pooled with learned per-n-gram weights (a learned
    IDF) and passed through an MLP head. Outputs: a 128-d joint embedding plus 64-d name-only and
    address-only embeddings.
  - Training: InfoNCE in both directions on 3.8M (Source-1, record) pairs of the non-held-out entities,
    batch 4096, country-homogeneous batches, masking of other matches of the same entity (no false
    negatives), 2 epochs of in-batch negatives then 2 epochs with **mined hard negatives** (the model's own
    top wrong neighbours). On held-out entities the true entity is the nearest neighbour for 96.9% of
    records and within the top-20 of some view for 99.65%.
  - Search: every Source-2/3 record queries the Source-1 entities **of its own country** with an exact GPU
    top-k (max over groups of 32 scores, then top-k inside the best groups: identical to brute force).
    Searching from the record side bounds the candidate list: each record contributes to a few entities.
  - Policy (chosen automatically on held-out entities): joint-view top-5 within 0.05 cosine of the
    record's best, plus address-view top-2 within 0.05 and above 0.80 (rescues records whose name is
    unrelated: brand names, acronyms), plus, **only for records without an address**, name-view top-5
    within 0.10 (the name view is searched only for those records). A **per-entity cap of 15 per view**
    stops hub entities from collecting thousands of candidates. The policy is the smallest candidate set
    whose F0.5 ceiling is within 0.0005 of the best of 216 policies (vectorised sweep, seconds); the cap is
    the smallest costing < 0.0002.
- **Candidate pairs generated:** 15,064,731 for the test set (**8.7 per Source-1 entity**; the test set has
  5.75 Source-2/3 records per Source-1 entity vs 4.7 in training). Held-out validation: 6.6 candidates per
  entity, p99 = 17, max = 39.
- **How you ensured true matches were not lost:** pair recall 99.0% on held-out entities, i.e. an F0.5
  ceiling of 0.9970 for a perfect matcher; the address view recovers name-mismatch records and the
  address-less name view recovers records with no address; the policy search trades candidate count
  against the F0.5 ceiling rather than raw recall.

---

## 4. Matching Model

**Features used (~150 per candidate pair):**
- Embedding: cosine in joint / address / name views, the pair's rank in the record's neighbour list per
  view, the record's best and second-best score per view, gap to best, lead over the best *other* entity.
- Name features: rapidfuzz ratio / token-sort / token-set / partial on the name, the core name, the cleaned
  core name and the skeleton; Jaro-Winkler; glued (space-free) partial ratio for domains/concatenations;
  acronym checks; **token accounting**: Source-1 core tokens with no fuzzy match in the record (a swapped
  word -> sibling) and vice versa (added filler -> noise), on raw and cleaned cores.
- Legal form: normalised legal-form sets of both sides, overlap and conflict ("LLC" vs "P.C.").
- Address features: fuzzy ratios on the normalised and the cleaned address; exact flags; address words
  missing from the record; **numbers** (from the cleaned address): counts, Jaccard, conflict, numbers on one
  side only, first-number equality, suffix relation (dropped leading digit), log absolute difference of the
  house numbers (siblings are close but not equal), last-two-digit agreement; empty-address flags.
- Name frequency: Source-1 entities of the country sharing the core name / skeleton of either side.
- Noise markers of the record (duplicated words, accents, OCR digits, leading junk, brackets, Indic
  script, domain name, upper case, zero padding, '#', PO box).
- Competition: each key similarity minus the best value among the record's OTHER candidates; exact-name
  candidate counts per record; exact-name candidates per entity over the full candidate set.
- Context: candidates per entity / per record, the record's rank among its entity's candidates.
- **Country is deliberately not a feature.**
- Stage 2 (on out-of-fold stage-1 probabilities): the record's best probability for another entity, the
  pair's rank within the record and within the entity, the entity's sum/max of probabilities, number of
  confident candidates.

**Model type:** XGBoost (hist, GPU, leaf-wise 127 leaves, learning rate 0.06, early stopping) for stage 1
and stage 2 (63 leaves); a single model refit on all held-out rows is applied to the test set. The code
also supports LightGBM on CPU and an MLP blend; on full-scale tests LightGBM and XGBoost were equal
(0.98665 vs 0.98666) and the MLP never won the blend. All models are trained from scratch, far below 8B
parameters (Apache-2.0 / MIT / BSD / ISC libraries).

**Threshold selection method:** decision rules are tuned on out-of-fold predictions of held-out entities:
(1) each record keeps only its highest-probability entity; (2) per entity, the prefix of its records that
maximises expected F0.5, `1.25*sum_top_m(p) / (m + 0.25*sum_all(p))`, or an empty list when
`P(no match) = prod(1-p)` is higher; alternatives (plain thresholds) are grid-searched and the best
out-of-fold macro F0.5 wins (here: expected-F0.5, alpha 1.0).

---

## 5. Results & Error Analysis

| Version | Main change | Validation F0.5 | Test candidates / entity |
|---|---|---|---|
| v1 | bi-encoder blocking, LightGBM + stacker | 0.9856 | 10.6 |
| v2 | name-frequency + competition features, address-less name view, hub cap | 0.9860 | 8.1 |
| **v3 (submitted)** | cleaned-text / legal-form / noise-marker features, 50% entity holdout (5x more matcher data), XGBoost GPU | **0.9880** | **8.7** |

- **F_0.5 Score (macro):** **0.9880** on 1,104,371 held-out Source-1 entities (India 0.9877, US 0.9882;
  singletons 0.9852); 0.9880 on the fixed 10% subset comparable with v2 (0.9860). Stage 1 alone: 0.9873.
- **Public leaderboard:** **0.9821** (test subset). The gap to validation (0.9880) is consistent with two
  shifts the validation cannot measure: France (15% of the test entities, absent from training, with very
  generic and repetitive institution names such as "Ecole", "Lycée", "Amicale", "Centre Hospitalier") and a
  denser test set (5.75 Source-2/3 records per Source-1 entity vs 4.7 in training, i.e. more decoys per
  entity). If US and India score at their validation level, France is around 0.95.
- **Blocking:** 99.0% pair recall, F0.5 ceiling 0.9970.
- **Realistic ceiling:** a perfect matcher that still loses the pairs blocking misses (0.95% of true pairs)
  and the genuinely ambiguous address-less records whose name is shared by several entities scores about
  **0.994** on the same data.
- **Where the remaining loss is (v2 decomposition, 220K entities):** entities with some missed matches
  51.6% of the loss; some wrong merges 22.8%; entities with matches predicted empty 15.8%; singletons given
  a false match 9.4%.
- **Singleton analysis:** the empty / non-empty decision is 25% of the loss. 89.5% of singleton false
  matches are decoy records that are near-exact noisy copies of the singleton entity (same address,
  typo-level name changes, median probability 0.76), and most wrongly empty entities are address-less
  records with shared names. An entity-level gate (AUC 0.9994 for "has any match") and a specialist model
  for single-candidate entities added only +0.00001 / +0.00014, so they were not adopted; the v3 features
  still lifted singleton F0.5 from 0.9766 to 0.9852.
- **Common false positives (wrong merges):** sibling distractors at the same street with a nearby house or
  unit number and one name word changed; decoy copies of singleton entities.
- **Common false negatives (missed matches):** address-less records whose name is shared by several
  entities; true copies whose house number and a name word were both perturbed; true pairs not retrieved
  by blocking (1.0%).
- **Unseen country (France):** normalisation is language-agnostic (anyascii, French abbreviations and
  legal forms included) and country is not a feature. On the test set, France receives 3.31 matches per
  entity (US 3.39, India 3.38) and a similar share of confident candidates (61.6% vs 59.6% / 58.6%).

**Experiments that did not help (full-scale, held-out, exact metric):** bigger or depth-wise trees
(0.98666 for every variant), averages of four GBDT variants (+0.00015), per-entity assignment features
using the other records of the same entity (+0.0002), an entity-level singleton gate (+0.00001), a
single-candidate specialist (+0.00014).

---

## 6. Conclusion

A learned, multi-view blocking stage reduces the search from 2.2M x 10.3M comparisons to under 9
candidates per entity at 99.0% recall, and a feature-rich GBDT with an assignment constraint and an
F0.5-optimal decision rule turns them into precise matches (validation F0.5 0.9880). The largest gains came
from understanding the noise process (cleaned text, legal forms, noise markers, name uniqueness) and from
giving the matcher entities the encoder never saw; the remaining errors are dominated by decoys and
shared names that the data cannot disambiguate.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/pipeline.py`: CLI. End to end (resumable):
  `BER_HOLDOUT_PCT=50 BER_MLP=0 python pipeline.py run-all --data <dataset> --work <tmp> --models <models> --out <output>`
  regenerates `matching_results.tsv` and `candidate_pairs.tsv` (plus `metrics.json`).
- `src/ber/`: `data.py`, `normalize.py`, `prepare.py` (stage 1); `encoder.py`, `train_encoder.py` (stage 2);
  `knn.py`, `blocking.py` (stage 3); `features.py` (stage 4); `matcher.py` (stage 5); `decide.py` (stage 6);
  `run.py` (orchestration, metrics, TSV writing, error analysis).
- `src/kaggle/`: the exact Kaggle job scripts used for the GPU runs (free T4, 29 GB RAM);
  `kernel_final` + `kernel_resume` produced the submitted v3.
- `README.md`, `requirements.txt` (pinned).

### B. Additional Results

Bi-encoder (v3, trained on 50% of entities) on held-out entities, searching all Source-1 entities of the
same country:

| Epoch | R@1 | R@5 | R@20 | any-view R@20 |
|---|---|---|---|---|
| 1 (in-batch negatives) | 0.9619 | 0.9794 | 0.9882 | 0.9962 |
| 2 | 0.9655 | 0.9821 | 0.9901 | 0.9964 |
| 3 (hard negatives) | 0.9682 | 0.9838 | 0.9910 | 0.9965 |
| 4 (hard negatives) | 0.9688 | 0.9843 | 0.9913 | 0.9965 |

Learning curve of the stage-1 matcher (full scale): 0.9831 / 0.9840 / 0.9849 with 25% / 50% / 100% of
the 10%-holdout training rows (+0.0009 per doubling), which motivated the 50% entity holdout.

Runtime on one Kaggle T4: encoder 42 min; training search 36 min; training features 10 min; matcher and
stacker about 25 min; test search about 25 min; test features and inference about 20 min.
