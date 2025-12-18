from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(expected, qs))
    if len(edges) < 3:
        return 0.0

    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)

    e = e_counts / max(1, e_counts.sum())
    a = a_counts / max(1, a_counts.sum())

    eps = 1e-9
    e = np.clip(e, eps, 1.0)
    a = np.clip(a, eps, 1.0)

    return float(np.sum((a - e) * np.log(a / e)))


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample KS statistic (no p-value; keeps dependencies light)."""
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")

    x = np.sort(x.astype(float))
    y = np.sort(y.astype(float))

    # union CDF
    all_vals = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, all_vals, side="right") / len(x)
    cdf_y = np.searchsorted(y, all_vals, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def drift_summary(
    features: pd.DataFrame,
    scores: dict[str, np.ndarray],
    split_idx: int,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}

    left = features.iloc[:split_idx]
    right = features.iloc[split_idx:]

    for col in features.columns:
        lx = left[col].to_numpy(dtype=float)
        rx = right[col].to_numpy(dtype=float)
        out[f"feature::{col}"] = {
            "psi": psi(lx, rx, bins=10),
            "ks": ks_statistic(lx, rx),
        }

    for name, s in scores.items():
        s = np.asarray(s, dtype=float)
        out[f"score::{name}"] = {
            "psi": psi(s[:split_idx], s[split_idx:], bins=10),
            "ks": ks_statistic(s[:split_idx], s[split_idx:]),
        }

    return out
