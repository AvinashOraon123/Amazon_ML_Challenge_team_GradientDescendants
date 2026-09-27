# Submission history

Each folder keeps the pipeline's own `metrics.json` for one submission (validation F0.5, blocking
statistics, decision rule, test statistics per country).

| Version | Validation F0.5 | Test candidates per entity | Documents |
|---|---|---|---|
| v1 | 0.9856 | 10.6 | `v1/metrics.json` |
| v2 | 0.9860 | 8.1 | `v2/METHODOLOGY_v2.md`, `v2/metrics.json` |
| v3 (submitted, leaderboard 0.9821) | 0.9880 | 8.7 | `../METHODOLOGY.md`, `v3/metrics.json` |

The step-by-step story of how the score moved between versions is in `../EXPERIMENTS.md`.

Note on `country_transfer` in `v3/metrics.json`: those two numbers (about 0.47) are an artifact of the
record subsampling used for that check in v3, not a model result; see `../METHODOLOGY.md`.
