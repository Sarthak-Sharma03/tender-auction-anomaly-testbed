from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BudgetMetrics:
    model: str
    budget: float
    flagged_sessions: int
    risky_sessions_flagged: int
    risky_sessions_total: int
    failed_events_flagged: int
    failed_events_total: int
    base_rate: float
    lift: float
    precision: float
    recall: float
    precision_ci_low: float
    precision_ci_high: float
    recall_ci_low: float
    recall_ci_high: float


@dataclass(frozen=True)
class BudgetThresholdEval:
    model: str
    budget: float
    threshold: float
    flagged_sessions_test: int
    flagged_rate_test: float
    risky_sessions_flagged_test: int
    risky_sessions_total_test: int
    failed_events_flagged_test: int
    failed_events_total_test: int
    base_rate_test: float
    lift_test: float
    precision_test: float
    recall_test: float


def proxy_risky_session(sessions: pd.DataFrame) -> pd.Series:
    # proxy label
    return pd.Series(sessions["failed_events"]).fillna(0).astype(float) > 0.0


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    k = max(0, min(int(k), int(n)))
    n = float(n)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z * np.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return float(lo), float(hi)


def compute_budget_metrics(
    sessions: pd.DataFrame,
    score: pd.Series,
    budgets: Iterable[float],
    model: str,
) -> list[BudgetMetrics]:
    """Budget-style evaluation for ranked anomaly scores.

    This is unsupervised. For quickstart we use a simple proxy label (`failed_events > 0`)
    so the evaluation harness is concrete and inspectable.

    Replace `proxy_risky_session` with a real label when you have one.
    Precision/recall intervals use Wilson bounds.
    """
    sdf = sessions.copy()
    sdf = sdf.assign(_score=pd.Series(score).to_numpy())
    sdf = sdf.sort_values("_score", ascending=False).reset_index(drop=True)

    risky_mask = proxy_risky_session(sdf)
    risky_total = int(risky_mask.sum())
    failed_total = int(pd.Series(sdf["failed_events"]).fillna(0).sum())
    base_rate = float(risky_total / len(sdf)) if len(sdf) > 0 else 0.0

    out: list[BudgetMetrics] = []

    for b in budgets:
        b = float(b)
        k = int(max(1, round(len(sdf) * b)))
        flagged_df = sdf.iloc[:k]
        flagged_risky_mask = proxy_risky_session(flagged_df)

        risky_flagged = int(flagged_risky_mask.sum())
        failed_flagged = int(pd.Series(flagged_df["failed_events"]).fillna(0).sum())

        precision = float(risky_flagged / k) if k > 0 else 0.0
        recall = float(risky_flagged / risky_total) if risky_total > 0 else 0.0
        lift = float(precision / base_rate) if base_rate > 0 else float("nan")
        prec_lo, prec_hi = wilson_interval(risky_flagged, k)
        rec_lo, rec_hi = wilson_interval(risky_flagged, risky_total)

        out.append(
            BudgetMetrics(
                model=str(model),
                budget=b,
                flagged_sessions=k,
                risky_sessions_flagged=risky_flagged,
                risky_sessions_total=risky_total,
                failed_events_flagged=failed_flagged,
                failed_events_total=failed_total,
                base_rate=base_rate,
                lift=lift,
                precision=precision,
                recall=recall,
                precision_ci_low=prec_lo,
                precision_ci_high=prec_hi,
                recall_ci_low=rec_lo,
                recall_ci_high=rec_hi,
            )
        )

    return out


def _threshold_for_budget(scores_train: np.ndarray, budget: float) -> float:
    scores_train = np.asarray(scores_train, dtype=float)
    scores_train = scores_train[np.isfinite(scores_train)]
    if len(scores_train) == 0:
        return float("nan")

    b = float(budget)
    b = max(1e-6, min(b, 1.0))
    k = int(max(1, round(len(scores_train) * b)))
    idx = np.argpartition(-scores_train, kth=k - 1)[k - 1]
    return float(scores_train[idx])


def threshold_eval(
    sessions_train: pd.DataFrame,
    score_train: np.ndarray,
    sessions_test: pd.DataFrame,
    score_test: np.ndarray,
    budgets: Iterable[float],
    model: str,
) -> list[BudgetThresholdEval]:
    """Pick a threshold on the train window to match a budget, apply it to the later window."""
    score_train = np.asarray(score_train, dtype=float)
    score_test = np.asarray(score_test, dtype=float)

    risky_test_mask = proxy_risky_session(sessions_test)
    risky_total_test = int(risky_test_mask.sum())
    failed_total_test = int(pd.Series(sessions_test["failed_events"]).fillna(0).sum())
    test_total = len(sessions_test)
    base_rate_test = float(risky_total_test / test_total) if test_total > 0 else 0.0

    out: list[BudgetThresholdEval] = []

    for b in budgets:
        thr = _threshold_for_budget(score_train, float(b))
        flagged_mask = score_test >= thr
        flagged_test = int(np.sum(flagged_mask))
        flagged_rate_test = float(flagged_test / test_total) if test_total > 0 else 0.0

        flagged_sessions = sessions_test.loc[flagged_mask]
        risky_flagged_test = int(proxy_risky_session(flagged_sessions).sum())
        failed_flagged_test = int(pd.Series(flagged_sessions["failed_events"]).fillna(0).sum())

        precision_test = float(risky_flagged_test / flagged_test) if flagged_test > 0 else 0.0
        recall_test = float(risky_flagged_test / risky_total_test) if risky_total_test > 0 else 0.0
        lift_test = float(precision_test / base_rate_test) if base_rate_test > 0 else float("nan")

        out.append(
            BudgetThresholdEval(
                model=str(model),
                budget=float(b),
                threshold=float(thr),
                flagged_sessions_test=flagged_test,
                flagged_rate_test=flagged_rate_test,
                risky_sessions_flagged_test=risky_flagged_test,
                risky_sessions_total_test=risky_total_test,
                failed_events_flagged_test=failed_flagged_test,
                failed_events_total_test=failed_total_test,
                base_rate_test=base_rate_test,
                lift_test=lift_test,
                precision_test=precision_test,
                recall_test=recall_test,
            )
        )

    return out
