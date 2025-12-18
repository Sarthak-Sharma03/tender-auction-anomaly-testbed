# Quickstart run

A small reproducible run meant to be skimmed (not a benchmark).

## Data
- synthetic generator (repo-contained)
- events: 151369
- sessions: 24297
- train_sessions: 17008
- test_sessions: 7289
- time_split: 0.7
- random_state: 13

## Models
- Baselines: robust_z, isolation_forest, one_class_svm, local_outlier_factor
- Primary score used for ranking/reporting: **ensemble_fisher**

## Summary
- base_risk_rate: 0.2941
- auc_precision_budget: 0.0261

## Budget evaluation (ranked, proxy label)
We report precision/recall under a fixed *review budget* using failed-logins as a proxy risk signal.

|   budget |   flagged |   risky_flagged |   risky_total |   precision |   recall |   lift |   failed_flagged | precision_ci   | recall_ci      |
|---------:|----------:|----------------:|--------------:|------------:|---------:|-------:|-----------------:|:---------------|:---------------|
|    0.005 |       121 |              92 |          7146 |      0.7603 |   0.0129 |   2.59 |              139 | [0.677, 0.828] | [0.011, 0.016] |
|    0.01  |       243 |             159 |          7146 |      0.6543 |   0.0223 |   2.22 |              263 | [0.593, 0.711] | [0.019, 0.026] |
|    0.05  |      1215 |             578 |          7146 |      0.4757 |   0.0809 |   1.62 |              925 | [0.448, 0.504] | [0.075, 0.087] |

## Threshold transfer check (train -> test)
For each budget, we pick a score threshold on the train window and apply it to the later window. This is a quick sanity check that the ranking isn't only good in-sample.

|   budget |   threshold |   flagged_test |   flagged_rate |   risky_flagged_test |   risky_total_test |   precision_test |   recall_test |   lift_test |   failed_flagged_test |
|---------:|------------:|---------------:|---------------:|---------------------:|-------------------:|-----------------:|--------------:|------------:|----------------------:|
|    0.005 |     32.758  |             45 |         0.0062 |                   38 |               2823 |           0.8444 |        0.0135 |        2.18 |                    64 |
|    0.01  |     29.1107 |             99 |         0.0136 |                   75 |               2823 |           0.7576 |        0.0266 |        1.96 |                   150 |
|    0.05  |     20.5644 |            522 |         0.0716 |                  259 |               2823 |           0.4962 |        0.0917 |        1.28 |                   458 |

## Risk curve (proxy label)
Risk rate by score quantile; should trend upward if ranking is meaningful.

|   bin |   score_min |   score_max |   sessions |   risk_rate |
|------:|------------:|------------:|-----------:|------------:|
|     1 |       0.331 |       1.738 |       2430 |       0.158 |
|     2 |       1.738 |       2.784 |       2430 |       0.211 |
|     3 |       2.785 |       3.89  |       2429 |       0.225 |
|     4 |       3.89  |       5.218 |       2430 |       0.226 |
|     5 |       5.219 |       6.544 |       2429 |       0.303 |
|     6 |       6.544 |       8.202 |       2430 |       0.326 |
|     7 |       8.204 |      10.199 |       2430 |       0.344 |
|     8 |      10.199 |      12.748 |       2429 |       0.351 |
|     9 |      12.748 |      17.242 |       2430 |       0.401 |
|    10 |      17.243 |      71.576 |       2430 |       0.396 |

## Ensemble agreement
Jaccard overlap between the top-k sessions of different ensemble methods.

|   budget |   jaccard |   overlap |   topk |
|---------:|----------:|----------:|-------:|
|    0.005 |     0.766 |       105 |    121 |
|    0.01  |     0.742 |       207 |    243 |
|    0.05  |     0.71  |      1009 |   1215 |

## Drift check (time split)
A lightweight drift summary between early and late windows (PSI + KS statistic). Drift isn't automatically 'bad' - it's a signal to re-check thresholds and monitoring.

| item | psi | ks |
|---|---:|---:|
| feature::events | 0.2475 | 0.2128 | 
| feature::unique_ips | 0.2475 | 0.2128 | 
| feature::unique_hours | 0.2098 | 0.1952 | 
| feature::events_per_hour | 0.1495 | 0.1650 | 
| feature::unique_device_rate | 0.1441 | 0.1609 | 
| feature::unique_countries | 0.1384 | 0.1596 | 
| feature::unique_browser_rate | 0.1344 | 0.1506 | 
| feature::unique_country_rate | 0.1215 | 0.1474 | 

## Notes
- failed_events is used as a proxy label for a quick decision-metric sanity check; treat it as a signal, not ground truth.
- quickstart caps the fit window for speed; see docs/EXPERIMENTS.md for config-driven runs.
- precision/recall intervals use Wilson binomial bounds.
