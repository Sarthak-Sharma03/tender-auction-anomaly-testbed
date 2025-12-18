# Architecture

This repo is intentionally small, but it is structured so a reviewer can trace the data flow end-to-end.

Domain note: the public data is a login-like surrogate for a tender/auction anomaly pipeline. See
`docs/DOMAIN_CONTEXT.md` for the field mapping and NDA rationale.

## Data flow (quickstart)

1. Load events (`data/Login_Data.xlsx` or synthetic generator)
2. Validate schema (`validate_events`)
3. Sessionize to user-day sessions
4. Build numeric session features
5. Fit baselines on an early time window (with a cap for speed)
6. Score all sessions
7. Combine via an ensemble (default: calibrated p-value aggregation using Fisher's method; also supports `ensemble_rank`)
8. Write outputs: scores, top sessions, budget metrics + confidence bounds, drift PSI/KS, risk curve, plots, and a model bundle

## Model bundle

Quickstart also writes a `model_bundle/` folder:
- `metadata.json` (run settings + feature columns)
- `*.joblib` model artifacts
- `calibration.json` (quantile-CDF sketch of train-window scores for calibrated ensembling in the API)

This is used by the optional API service and makes the run inspectable without hidden state.
