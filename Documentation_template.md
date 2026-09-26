# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** GradientDescendants
**Team Members:** [fill in]
**Submission Date:** [fill in]

---

## 1. Executive Summary
A classic *block → retrieve → score → decide* entity-resolution pipeline that runs the full
dataset on a 2-vCPU / 8 GB machine. Country blocks plus sparse TF-IDF nearest-neighbour search
(over names, addresses **and their phonetic consonant skeletons**, which recover
phonetically-transliterated Indic names) produce ~15 candidates per S2/S3 record; a LightGBM
classifier on ~70 country-agnostic similarity, **competition** and **ambiguity** features scores
them, and a threshold tuned for macro F0.5 on held-out entities makes the final decision.

---

## 2. Methodology

### 2.1 Problem Analysis
- Every labelled true pair shares its `country` → country is a lossless hard block; the test-only
  country (France) simply forms its own block.
- Every S2/S3 record belongs to **at most one** S1 entity → retrieve from the S2/S3 side (its
  single true S1 only needs to be in its own top-K) and measure how clearly one S1 beats its
  rivals for the same record.
- ~43% of true pairs have different cleaned names (DBA / renamed entities, typos, website names,
  transliterations) but most share address tokens or house numbers → the address must drive
  retrieval as much as the name.
- ~4–5% of S2/S3 names are in Devanagari / Kannada / Telugu script; after transliteration they are
  *phonetic* spellings of English words ("brait prodyusr praivet" = "Bright Producer Private").
- Many S1 records share generic names ("family health", "grace chapel"); when an S2/S3 record has
  no address, only the name's rarity tells whether a name match is trustworthy.
- 5.6% of S1 entities are singletons; each false merge on a singleton costs a full point.

### 2.2 Solution Strategy
**Approach Type:** Blocking + nearest-neighbour candidate generation + gradient-boosted classifier
**Core Innovation:** phonetic-skeleton retrieval for transliterated names; competition and ambiguity
features computed from streaming retrieval statistics; a fully streamed pipeline that fits the
full data on a free-tier instance.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** `country` (hard block), then TF-IDF cosine top-15 inside the block on a
  combined vector: name char 3-grams ⊕ phonetic-name char 3-grams ⊕ address word 1–2-grams ⊕
  phonetic-address word 1–2-grams (equal weights; n-grams in > 20k S1 records dropped).
- **Two-stage candidate generation:** (1) retrieval of the top-15 S1 per S2/S3 record (~70–87 per S1),
  then (2) a **learned pruner** — a small LightGBM that sees only retrieval-stage signals (retrieval
  score split into name / address parts, rank, competition against the record's and the S1's
  other candidates, near-ties, name/address frequency) — keeps pairs with q ≥ 0.003. The survivors
  are exactly what the final matcher scores and what `candidate_pairs.tsv` contains.
- **Candidate pairs generated:** [fill in from 05] — on the full India validation block: 7.5 per S1
  entity (from 70.4), candidate recall 97.4% (from 97.5%), final F0.5 change −0.0002.
- **How we ensured true matches were not lost:** retrieval from the S2/S3 side; phonetic parts
  (+0.8 pt recall@10 on the full India block); K chosen from the recall-vs-K curve; blocking and
  candidate recall reported in notebook 02 ([fill in]%).

---

## 4. Matching Model

**Features used:**
- Name features: rapidfuzz ratio / partial / token-sort / token-set / Jaro-Winkler, full-name
  token set, no-space ratio (web domains), alias-aware best score (fka/dba), TF-IDF char cosine,
  token Jaccard / overlap, first-token and exact equality, length ratio; phonetic-skeleton ratio /
  token set / no-space ratio; legal-form Jaccard.
- Address features: ratio / partial / token-sort / token-set, phonetic token set, TF-IDF word
  cosine, Jaccard, containment (partial addresses), numeric-token Jaccard / shared / conflict,
  first-number equality, postcode and state agreement, empty flags.
- Other: retrieval score and its name / address parts and rank; competition features per S2/S3
  record and per S1 entity (relative score, margin to the best rival, best-vs-second gap,
  near-tie count, candidate counts); ambiguity features (number of S1 in the block sharing the
  exact name / address).

**Model type:** LightGBM binary classifier (127 leaves, lr 0.08, early stopping) trained on 30% of S1 entities; easy negatives (retrieval score < 0.8x the record's best) subsampled to 10% with weight 10
**Threshold selection method:** macro F0.5 (leaderboard definition, every entity incl. singletons)
maximised on valid-A entities; reported on untouched valid-B entities.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, valid-B):** [fill in]
- **Common false positives (wrong merges):** records without an address carrying a common name
  shared by several S1 businesses; near-identical distractor records (same name, house number
  differing by a few digits).
- **Common false negatives (missed matches):** renamed / random-name records with a truncated
  address; heavily garbled phonetic transliterations.

---

## 6. Conclusion
Careful blocking plus rich relative ("how much better than the rivals") features let a single
gradient-boosted model reach high precision; phonetic keys close most of the transliteration gap,
and streaming every stage makes the full pipeline reproducible on free-tier hardware.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `src/ber/` (package), `notebooks/00…05` (run in order, or
`scripts/run_notebooks.sh`), `requirements.txt`, `README.md`. Notebook 05 writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs the official validator.

### B. Additional Results
See the executed notebooks (block report, recall-vs-K, feature separation plots, PR curve,
F0.5-vs-threshold curve, feature importance, error examples).
