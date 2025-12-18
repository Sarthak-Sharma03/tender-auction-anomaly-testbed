from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def rank_ensemble(
    scores: dict[str, np.ndarray],
    members: Iterable[str] | None = None,
) -> np.ndarray:
    """Rank-ensemble: each member contributes normalized ranks (0..1), then averaged."""
    members = list(members) if members is not None else list(scores.keys())
    if not members:
        raise ValueError("No members for ensemble.")
    n = len(next(iter(scores.values())))
    agg = np.zeros(n, dtype=float)

    for name in members:
        s = np.asarray(scores[name], dtype=float)
        if len(s) != n:
            raise ValueError("All score vectors must have the same length.")
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        if n > 1:
            ranks = ranks / (n - 1)
        agg += ranks

    return agg / len(members)


def empirical_pvalues(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """One-sided empirical p-values for anomaly scores (higher = more anomalous).

    p(s) ~= P_train(S >= s). Uses add-one smoothing for stability.
    """
    train = np.asarray(train_scores, dtype=float)
    x = np.asarray(scores, dtype=float)

    train = train[~np.isnan(train)]
    if train.size == 0:
        return np.ones_like(x, dtype=float)

    ts = np.sort(train)
    idx = np.searchsorted(ts, x, side="left")
    count_ge = ts.size - idx
    p = (count_ge.astype(float) + 1.0) / (ts.size + 1.0)

    # NaN safe
    return np.where(np.isnan(x), 1.0, p)


def fisher_stat(pvals: np.ndarray) -> np.ndarray:
    """Fisher combination statistic for p-values.

    This returns the chi-square test statistic:  -2 * sum log(p_i).
    For ranking purposes, the statistic itself is sufficient (higher => more anomalous).
    """
    p = np.asarray(pvals, dtype=float)
    p = np.clip(p, 1e-300, 1.0)
    return -2.0 * np.sum(np.log(p), axis=0)


def fisher_ensemble(
    scores: dict[str, np.ndarray],
    train_scores: dict[str, np.ndarray],
    members: Iterable[str] | None = None,
) -> np.ndarray:
    """Calibrated ensemble: convert member scores to empirical p-values (train window),
    then combine via Fisher's method.

    This tends to be more stable under score-scale mismatches than a raw-score average.
    """
    members = list(members) if members is not None else list(scores.keys())
    if not members:
        raise ValueError("No members for ensemble.")

    pcols = []
    n = None
    for name in members:
        s = np.asarray(scores[name], dtype=float)
        t = np.asarray(train_scores[name], dtype=float)
        if n is None:
            n = len(s)
        if len(s) != n:
            raise ValueError("All score vectors must have the same length.")
        pcols.append(empirical_pvalues(t, s))

    return fisher_stat(np.vstack(pcols))
