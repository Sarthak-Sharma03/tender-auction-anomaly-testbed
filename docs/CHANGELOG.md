# Changelog

## 0.7.0
- Added risk diagnostics (risk curve, ensemble agreement) and budget lift with confidence bounds.
- Expanded session features (temporal diversity and rate features) for richer anomaly signals.
- Report now includes summary stats (base risk rate, AUC over budget curve) and new tables.

## 0.6.1
- Reframed documentation around tender/auction domain with NDA-safe surrogate mapping.
- Expanded technical report with related work, assumptions, and synthetic results/ablations.
- Added reviewer guidance updates and experiment notes for ablations.

## 0.6.0
- Added calibrated ensemble (`ensemble_fisher`): empirical p-values (train window) combined via Fisher's method.
- Wrote `calibration.json` into the model bundle so the FastAPI scorer reproduces calibrated aggregation.
- Fixed report table truncation by rendering markdown tables without pandas display ellipsis.
- Added paper-style narrative: `docs/TECHNICAL_REPORT.md` (and `docs/TECHNICAL_REPORT.pdf`).
- Added config support for `ensemble_method` and `synthetic_cfg` (tune synthetic drift/anomaly rate without code).

## 0.5.0
- Added train->test threshold transfer check and output file.
- Drift summary now includes KS statistic alongside PSI.
- Added Makefile and LICENSE.

## 0.4.0
- Fixed packaging (valid pyproject) and stabilized CLI commands.
- Added synthetic generator for fully reproducible runs without private data.
- Added plots + Markdown report writer.
- Added model bundle output and optional FastAPI scorer (`tad serve-api`).
- Added YAML config runner (`tad run-config`) and example configs.

## 0.2.0
- Baseline suite (IsolationForest / OneClassSVM / LOF) and rank-ensemble.
- Modularized feature extraction, scoring, and evaluation.
