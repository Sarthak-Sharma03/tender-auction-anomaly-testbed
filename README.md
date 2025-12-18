# tender-auction-anomaly-testbed

Decision-centric anomaly detection testbed for tender/auction event streams, packaged as a reviewer-friendly ML systems artifact.

NDA note: the production tender/auction data and rules are under NDA. This public repo uses an anonymized, login-like
surrogate dataset plus a synthetic generator so the full pipeline stays reproducible. See `docs/DOMAIN_CONTEXT.md` for
the field mapping and rationale.

Start here (resume readers):
- `docs/TECHNICAL_REPORT.pdf` (paper-style narrative + results)
- `docs/REVIEWER_GUIDE.md` (fast skim)
- `docs/DOMAIN_CONTEXT.md` (NDA-safe mapping)

## What this artifact shows

- End-to-end pipeline: load -> validate -> sessionize -> feature matrix -> unsupervised baselines -> ensemble -> reports
- Decision-centric evaluation under fixed review budgets (precision/recall at budget + confidence bounds + lift)
- Time-split drift summary (PSI + KS) and train -> test threshold transfer
- Risk diagnostics (risk curve by score quantile, ensemble agreement)
- Reproducible tooling: package layout, CLI, tests, CI, config-driven runs
- Inspectable model bundle and a tiny reference scoring API (FastAPI)

## Quickstart (2-5 minutes)

```bash
python -m pip install -e ".[dev]"
pytest -q
tad quickstart --out-dir reports/quickstart  # default ensemble=fisher (calibrated)
```

Outputs land in `reports/quickstart/` (report, metrics, drift, plots, and a model bundle).

### No data? Use synthetic

```bash
tad quickstart --synthetic --out-dir reports/quickstart_synthetic
```

## Experiments and ablations

```bash
tad run-config experiments/configs/quickstart_sample.yaml
tad run-config experiments/configs/quickstart_synthetic.yaml
tad run-config experiments/configs/quickstart_synthetic_high_drift.yaml
tad run-config experiments/configs/quickstart_synthetic_rank.yaml
```

See `docs/EXPERIMENTS.md` for config notes and the ablation set used in the report.

## Reference API (optional)

```bash
python -m pip install -e ".[api]"
tad serve-api --bundle reports/quickstart/model_bundle --port 8000
```

## Notes

- Quickstart uses a proxy risk signal so the evaluation harness is concrete; see `docs/TECHNICAL_REPORT.md`.
- The goal is an inspectable workflow and engineering quality, not a claim of SOTA accuracy.

## Legacy material

The original demo/UI and older artifacts are preserved under `legacy/`.
