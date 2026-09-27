# ML Challenge 2026: Business Entity Resolution Solution Template (submission v2)

**Team Name:** Gradient_Descendants
**Team Members:** Avinash Oraon, Kshirod Kalet, Samprit Halder, Ayush Chanderia
**Submission Date:** 2026-09-27

> This is the methodology document of **submission v2** (validation F0.5 0.9860), kept for the record.
> The submitted final version is v3 (validation F0.5 0.9880, leaderboard 0.9821): see
> `solution_v3/METHODOLOGY.md` and `solution_v3/EXPERIMENTS.md`. v2's own metrics are in
> `metrics.json` next to this file.

---

## 1. Executive Summary

We treat entity resolution as *"which Source-1 entity does each Source-2/3 record belong to, if any?"*.
A hashed character n-gram **bi-encoder** trained from scratch (contrastive loss with mined hard
negatives) produces joint, address-only and name-only embeddings; **multi-view GPU nearest-neighbour
search** with a per-entity hub cap yields ~6.1 candidates per Source-1 entity while keeping 99.05% of
true pairs. A **LightGBM + MLP pair classifier** on ~110 embedding, fuzzy-string, number, token,
name-frequency and competition features, a **stage-2 stacker** over competing candidates, a
**one-entity-per-record** constraint and an **expected-F0.5 decision rule** produce the final matches.
Held-out validation macro F0.5: **0.9860**.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from exploring the 2.2M / 5.0M / 5.3M training records and 7.64M labelled pairs:

- **No true pair crosses countries**, so every search runs within one country (an open set of labels; France
  in the test set is handled by the same code path with no country-specific branches).
- **Each Source-2/3 record matches at most one Source-1 entity** (verified on all 7.64M pairs). This turns
  the problem into a per-record assignment and is exploited both in blocking and in the final decision.
- 5.6% of Source-1 entities have no match (singletons); matched entities have 1-11 matches (mode 3).
- 26% of Source-2/3 records (2.68M) match nothing: they act as **distractors**, many of them deliberate
  **siblings** of a real entity: same street with a nearby house number (544 vs 557 Portofino Loop,
  8-2-603/M/19 vs /M/24) and a name with one word swapped ("Wright Horizon Equipment" vs "... Petroleum").
- Name noise: word shuffles, legal suffix moves/drops ("Inc Corpfilbdte Zeo"), OCR digits ("6eneral",
  "We1lness"), spurious accents, domains ("indriyaclub.com"), acronyms ("SHLA" for "Smith, Holman and
  Love Alpex"), unrelated brand names at the true address ("Onyxxylo"), and 11-15% of names in Indic
  scripts (Devanagari, Gujarati, Bengali, Kannada...).
- Address noise: case, abbreviations (RD/Road, R./Rue), zero padding ("0701"), dropped leading digits
  ("476" for 7476), state name vs code, native-script state names, "null" tokens, component reordering,
  3.3% empty addresses.

### 2.2 Solution Strategy

**Approach Type:** Hybrid: learned blocking (bi-encoder + multi-view kNN) + pairwise classifier + stacked
assignment/decision layer.
**Core Innovation:** GPU-hashed character n-gram bi-encoder with three output views (joint / address /
name) searched with an exact grouped top-k, combined with a record-to-entity assignment constraint and
an expected-F0.5 decision rule.

Pipeline (all stages resumable; code in `code/business_entity_resolution/src/ber/`):

1. **Normalisation** (`normalize.py`, polars, rule-based and country-agnostic): transliteration with
   `anyascii`, lowercasing, accent/punctuation removal, domain unwrapping, junk reference removal,
   legal-form stripping to a *core name*, a **consonant skeleton** of the core name (bridges
   transliteration: "constructions" -> `knstrktns`, Gujarati -> `knstrksns`), address abbreviation
   expansion (incl. French), zero-padding removal, state name -> code mapping.
2. **Bi-encoder** (`encoder.py`, `train_encoder.py`).
3. **Candidate generation** (`knn.py`, `blocking.py`).
4. **Pair features** (`features.py`).
5. **Matcher** (`matcher.py`): LightGBM + MLP, stage-2 stacker.
6. **Decision** (`decide.py`).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** no hand-made keys; learned embeddings.
  - Bi-encoder: each record's normalised name, skeleton and address are fixed-width byte arrays; character
    2-5-grams are **hashed on the GPU** into 4M buckets, pooled with learned per-n-gram weights (a learned
    IDF) and passed through an MLP head. Outputs: a 128-d joint embedding plus 64-d name-only and
    address-only embeddings.
  - Training: InfoNCE in both directions on 6.9M (Source-1, record) pairs, batch 4096, country-homogeneous
    batches, masking of other matches of the same entity (no false negatives), 2 epochs of in-batch
    negatives then 2 epochs with **mined hard negatives** (the model's own top wrong neighbours).
    Held-out (never-trained) entities: the true entity is the nearest neighbour for 97.5% of records and
    in the top-20 of some view for 99.67%.
  - Search: every Source-2/3 record queries the Source-1 entities **of its own country** in each view with
    an exact GPU top-k (grouped max-reduction + top-k inside the best groups, identical to brute force).
    Searching from the record side bounds the candidate list: each record contributes to at most a few
    entities.
  - Policy (chosen automatically on the validation holdout): keep the joint-view top-3 neighbours within
    0.10 cosine of the record's best, plus the address-view best neighbour if its cosine is above 0.80
    (rescues records whose name is unrelated: brand names, acronyms), plus, **only for records without an
    address**, the name-view top-5 within 0.10 of the best (62% of the misses of the first version were
    address-less records). A **per-entity cap of 15 per view** keeps hub entities from collecting
    thousands of candidates. The policy is the smallest candidate set whose F0.5 recall ceiling is within
    0.0005 of the best of 216 policies; the cap is the smallest that costs < 0.0002.
- **Candidate pairs generated:** 14,094,563 for the test set (**8.1 per Source-1 entity**; the test set has
  5.75 Source-2/3 records per Source-1 entity vs 4.7 in training). On the validation holdout: 6.1 candidates
  per entity, p99 = 15, max = 35.
- **How you ensured true matches were not lost:** pair recall 99.05% on held-out entities, i.e. an F0.5
  ceiling of 0.9971 for a perfect matcher; the address view recovers most name-mismatch records; the
  policy search explicitly trades candidate count against the F0.5 ceiling rather than raw recall.

---

## 4. Matching Model

**Features used (~110 per candidate pair):**
- Embedding: cosine in joint / address / name views, the pair's rank in the record's neighbour list per
  view, the record's best and second-best score per view, gap to best, lead over the best *other* entity.
- Name features: rapidfuzz ratio / token-sort / token-set / partial ratio on the name, the core name and
  the skeleton; Jaro-Winkler on the core name; glued (space-free) partial ratio for domains/concatenations;
  acronym checks (initials vs. glued name); **token accounting**: number/fraction of Source-1 core tokens
  with no fuzzy match in the record (a swapped word -> sibling) and vice versa (added filler words ->
  noise).
- Address features: ratio / token-sort / token-set / partial ratio; exact-address flag; **numbers**:
  counts, Jaccard, conflict flag, numbers present on one side only, first-number equality, suffix relation
  (dropped leading digit), log absolute difference of the house numbers (siblings are close but not equal),
  last-two-digit agreement; empty-address flags.
- Name frequency: how many Source-1 entities of the country share the core name / skeleton of either side
  (an address-less record whose exact core name belongs to a single entity is that entity's in 97.7% of
  cases; shared names are genuinely ambiguous).
- Competition: each similarity (joint/address/name cosine, core/name/skeleton/address fuzzy scores, number
  and exact-name flags) minus the best value among the record's OTHER candidates; exact-name candidate
  counts per record and per entity.
- Context: candidates per entity / per record, the record's rank among its entity's candidates and gap to
  the entity's best, source (S2/S3). **Country is deliberately not a feature.**
- Stage 2 (stacking on out-of-fold stage-1 probabilities): the record's best probability for any other
  entity, rank of the pair within the record and within the entity, entity's sum/max of probabilities,
  number of confident candidates.

**Model type:** LightGBM (127 leaves, early stopping) and a 4-layer MLP (BatchNorm, GELU, dropout, GPU),
blended on out-of-fold predictions (LightGBM weight chosen: 1.0), followed by a LightGBM stage-2 stacker.
All models are trained from scratch (no pretrained weights; far below 8B parameters; MIT / BSD / ISC
licensed libraries).

**Threshold selection method:** trained and tuned only on candidates of the 10% held-out Source-1 entities
(never seen by the encoder), 5-fold cross-validation split by record so the assignment step sees
out-of-fold probabilities. Decision: (1) each record keeps only its highest-probability entity; (2) per
entity the prefix of its records maximising expected F0.5 (`1.25*sum_top_m(p) / (m + 0.25*sum_all(p))`,
empty list when `P(no match) = prod(1-p)` is higher). The rule, its parameters, the blend weight and the
use of stage 2 are selected by grid search on the out-of-fold macro F0.5.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9860** on 220,660 held-out Source-1 entities (India 0.9855, US 0.9864;
  singletons 0.9766). Stage 1 alone: 0.9850. First version (without name-frequency / competition features
  and the address-less name view): 0.9856.
- **Blocking:** 99.05% pair recall, F0.5 ceiling 0.9971, 6.1 candidates per entity (validation), 8.1 (test).
- **Error budget (first version, 762,235 held-out true pairs):** 3,436 wrong merges vs 24,840 missed pairs;
  62% of the missed pairs are address-less records; 34% of the misses were never retrieved by blocking.
- **Unseen-country proxy:** training the matcher on one country and testing on the other gives
  India->US 0.970 and US->India 0.956, so France (15% of the test set, absent from training) is the main
  risk; this is why country is not a feature and normalisation is language-agnostic.
- **Common false positives (wrong merges):** sibling distractors at the same street with a nearby house or
  unit number and one name word changed; generic names without an address that exist twice in Source 1
  ("Southern Montessori School").
- **Common false negatives (missed matches):** records with an empty address and a generic or heavily
  altered name; records whose name was replaced by an unrelated brand name and whose address is partial;
  true pairs not retrieved by blocking (1.1%).

---

## 6. Conclusion

A learned, multi-view blocking stage reduces the search from 2.2M x 10.3M comparisons to ~6-8 candidates
per entity at 99.05% recall, and a feature-rich pair classifier with an assignment constraint and an
F0.5-optimal decision rule turns them into precise matches. The biggest remaining gains are in separating
sibling distractors from true matches and in robustness to unseen countries.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/pipeline.py`: CLI; `python pipeline.py run-all --data <dataset> --work <tmp> --models <models> --out <output>`
  regenerates both output files end to end (resumable).
- `src/ber/`: `data.py`, `normalize.py`, `prepare.py` (stage 1); `encoder.py`, `train_encoder.py` (stage 2);
  `knn.py`, `blocking.py` (stage 3); `features.py` (stage 4); `matcher.py` (stage 5); `decide.py` (stage 6);
  `run.py` (orchestration, metrics, TSV writing, error analysis).
- `src/kaggle/`: the exact Kaggle job scripts used for the GPU runs (free T4, 29 GB RAM).
- `README.md`, `requirements.txt` (pinned).

### B. Additional Results

Bi-encoder on held-out entities (762,235 pairs, searching all 2.2M Source-1 entities of the same country):

| Epoch | R@1 | R@5 | R@20 | any-view R@20 |
|---|---|---|---|---|
| 1 (in-batch negatives) | 0.9692 | 0.9846 | 0.9917 | 0.9965 |
| 2 | 0.9712 | 0.9859 | 0.9924 | 0.9966 |
| 3 (hard negatives) | 0.9742 | 0.9875 | 0.9932 | 0.9966 |
| 4 (hard negatives) | 0.9751 | 0.9881 | 0.9935 | 0.9967 |

Runtime on one Kaggle T4: encoder training 70 min; train search 52 min (reused from a saved artifact in
later runs); matcher 35 min; test search 30 min; test features 9 min; test inference with a single full
LightGBM model ~20 min (averaging five fold models took 2.4 h on 4 CPU cores).
