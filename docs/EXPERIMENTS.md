# Experiments

Quickstart is a default run meant for review. For repeatable runs, use YAML configs.

```bash
tad run-config experiments/configs/quickstart_sample.yaml
tad run-config experiments/configs/quickstart_synthetic.yaml
tad run-config experiments/configs/quickstart_synthetic_high_drift.yaml
tad run-config experiments/configs/quickstart_synthetic_rank.yaml
```

A few practical notes:
- Budgets are expressed as fractions (0.01 = 1% of sessions flagged).
- The pipeline is unsupervised; proxy labels are for sanity-checking the evaluation harness.
- If you plug in a real label (fraud/abuse/incident), update the metric code rather than overfitting thresholds.

Outputs (per run):
- `budget_metrics.json` (precision/recall + lift + confidence bounds)
- `budget_threshold_eval.json` (train->test transfer)
- `risk_curve.json` (proxy risk rate by score quantile)
- `ensemble_agreement.json` (agreement between ensemble methods)
- `drift.json` (PSI + KS)

## Ablations used in the report

- Ensemble ablation: compare `ensemble_fisher` vs `ensemble_rank` on the same synthetic stream
  (`experiments/configs/quickstart_synthetic.yaml`).
- Drift stress test: increase anomaly rate + drift day (`experiments/configs/quickstart_synthetic_high_drift.yaml`).

## Config fields

- `ensemble_method`: `fisher` (default) or `rank`.
- `synthetic`: boolean.
- `synthetic_cfg`: optional dict passed into `SyntheticConfig` (tune drift/anomaly rate without code).
