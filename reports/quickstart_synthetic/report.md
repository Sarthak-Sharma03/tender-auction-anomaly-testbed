# Quickstart run

A small reproducible run meant to be skimmed (not a benchmark).

## Data
- synthetic generator (repo-contained)
- events: 34437
- sessions: 7608
- train_sessions: 5326
- test_sessions: 2282
- time_split: 0.7
- random_state: 13

## Models
- Baselines: robust_z, isolation_forest, one_class_svm, local_outlier_factor
- Primary score used for ranking/reporting: **ensemble_fisher**

## Summary
- base_risk_rate: 0.1951
- auc_precision_budget: 0.0318

## Budget evaluation (ranked, proxy label)
We report precision/recall under a fixed *review budget* using failed-logins as a proxy risk signal.

|   budget |   flagged |   risky_flagged |   risky_total |   precision |   recall |   lift |   failed_flagged | precision_ci   | recall_ci      |
|---------:|----------:|----------------:|--------------:|------------:|---------:|-------:|-----------------:|:---------------|:---------------|
|    0.005 |        38 |              32 |          1484 |      0.8421 |   0.0216 |   4.32 |               38 | [0.696, 0.926] | [0.015, 0.030] |
|    0.01  |        76 |              60 |          1484 |      0.7895 |   0.0404 |   4.05 |               98 | [0.685, 0.866] | [0.032, 0.052] |
|    0.05  |       380 |             226 |          1484 |      0.5947 |   0.1523 |   3.05 |              351 | [0.545, 0.643] | [0.135, 0.171] |

## Threshold transfer check (train -> test)
For each budget, we pick a score threshold on the train window and apply it to the later window. This is a quick sanity check that the ranking isn't only good in-sample.

|   budget |   threshold |   flagged_test |   flagged_rate |   risky_flagged_test |   risky_total_test |   precision_test |   recall_test |   lift_test |   failed_flagged_test |
|---------:|------------:|---------------:|---------------:|---------------------:|-------------------:|-----------------:|--------------:|------------:|----------------------:|
|    0.005 |     35.6461 |              8 |         0.0035 |                    5 |                577 |           0.625  |        0.0087 |        2.47 |                     7 |
|    0.01  |     27.8991 |             37 |         0.0162 |                   30 |                577 |           0.8108 |        0.052  |        3.21 |                    60 |
|    0.05  |     17.8111 |            187 |         0.0819 |                  116 |                577 |           0.6203 |        0.201  |        2.45 |                   188 |

## Risk curve (proxy label)
Risk rate by score quantile; should trend upward if ranking is meaningful.

|   bin |   score_min |   score_max |   sessions |   risk_rate |
|------:|------------:|------------:|-----------:|------------:|
|     1 |       0.133 |       1.8   |        761 |       0.038 |
|     2 |       1.801 |       2.811 |        761 |       0.101 |
|     3 |       2.812 |       4.184 |        761 |       0.164 |
|     4 |       4.184 |       5.697 |        760 |       0.213 |
|     5 |       5.697 |       6.818 |        761 |       0.152 |
|     6 |       6.818 |       8.071 |        761 |       0.181 |
|     7 |       8.072 |      10.02  |        760 |       0.172 |
|     8 |      10.02  |      13.013 |        761 |       0.269 |
|     9 |      13.014 |      14.73  |        761 |       0.152 |
|    10 |      14.746 |      66.447 |        761 |       0.506 |

## Ensemble agreement
Jaccard overlap between the top-k sessions of different ensemble methods.

|   budget |   jaccard |   overlap |   topk |
|---------:|----------:|----------:|-------:|
|    0.005 |     0.949 |        37 |     38 |
|    0.01  |     0.727 |        64 |     76 |
|    0.05  |     0.747 |       325 |    380 |

## Drift check (time split)
A lightweight drift summary between early and late windows (PSI + KS statistic). Drift isn't automatically 'bad' - it's a signal to re-check thresholds and monitoring.

| item | psi | ks |
|---|---:|---:|
| feature::events | 0.1264 | 0.1472 | 
| feature::unique_ips | 0.1264 | 0.1472 | 
| feature::unique_hours | 0.1052 | 0.1339 | 
| feature::events_per_hour | 0.0949 | 0.1129 | 
| feature::unique_countries | 0.0893 | 0.1296 | 
| feature::unique_device_rate | 0.0818 | 0.1119 | 
| feature::unique_browser_rate | 0.0557 | 0.0980 | 
| score::isolation_forest | 0.0484 | 0.0815 | 

## Notes
- failed_events is used as a proxy label for a quick decision-metric sanity check; treat it as a signal, not ground truth.
- quickstart caps the fit window for speed; see docs/EXPERIMENTS.md for config-driven runs.
- precision/recall intervals use Wilson binomial bounds.
