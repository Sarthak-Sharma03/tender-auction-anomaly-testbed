# Reviewer guide

This repo is designed to be skimmed. Start with `docs/TECHNICAL_REPORT.pdf` if you want a single narrative, and
`docs/DOMAIN_CONTEXT.md` for the tender/auction mapping + NDA note.

## Lens A - Decision-centric anomaly detection (research style)

Question: with a fixed human review budget, which sessions should be flagged first?

Run:

```bash
tad quickstart --out-dir reports/quickstart  # default ensemble=fisher
```

What to look at:
- `reports/quickstart/report.md`
- `reports/quickstart/budget_metrics.json` (precision/recall + confidence bounds + lift)
- `reports/quickstart/budget_threshold_eval.json` (threshold picked on train, applied to test)
- `reports/quickstart/risk_curve.json` (proxy risk rate by score quantile)
- `reports/quickstart/ensemble_agreement.json` (agreement between ensemble methods)
- `reports/quickstart/drift.json`
- `reports/quickstart/plots/`

Notes:
- This is unsupervised. Quickstart uses a proxy risk signal so the metrics are concrete.
- A synthetic run (`--synthetic`) reproduces the workflow without private data.

## Lens B - Systems / applied reference implementation

Question: is this pipeline reproducible and extendable into a reliable service?

Pointers:
- `src/tad/` contains the real implementation (package layout, CLI, tests).
- `experiments/configs/` shows config-driven runs and ablations.
- `tad serve-api` is a small FastAPI scorer using the model bundle produced by quickstart.

If you only read one file, read: `src/tad/quickstart.py`.
