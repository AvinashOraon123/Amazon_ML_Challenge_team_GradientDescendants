# Full-scale experiments

Scripts used for the analyses reported in `METHODOLOGY.md` (run from this folder against the
full training data and the feature table exported by the `analyze` stage):

- `exp_ceiling.py` realistic F0.5 ceiling (unreachable + genuinely ambiguous pairs)
- `exp_curve.py` learning curve of the stage-1 matcher (+0.0009 per doubling of training rows)
- `exp_newfeat.py` value of the cleaned-text / legal-form / noise-marker features (+0.0016)
- `exp_stage3.py` assignment features from the other records of an entity (+0.0002)
- `exp_tune.py`, `score_oofs.py` LightGBM vs XGBoost, bigger trees, averages (<= +0.00015)
- `exp_singletons.py` where F0.5 is lost (per-entity decomposition)
- `exp_gate.py` entity-level "has any match" gate (+0.00001)
- `exp_single_specialist.py` specialist for single-candidate entities (+0.00014)
- `prep_local.py` local preparation helper
