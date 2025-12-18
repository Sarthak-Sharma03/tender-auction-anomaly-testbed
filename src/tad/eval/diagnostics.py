from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .metrics import proxy_risky_session


def score_risk_curve(
    sessions: pd.DataFrame,
    scores: np.ndarray,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Risk rate by score quantile using the proxy label."""
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    if valid.sum() == 0:
        return pd.DataFrame(
            columns=["bin", "score_min", "score_max", "sessions", "risk_rate"]
        )

    s = scores[valid]
    sess = sessions.loc[valid].reset_index(drop=True)
    edges = np.unique(np.quantile(s, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return pd.DataFrame(
            columns=["bin", "score_min", "score_max", "sessions", "risk_rate"]
        )

    bin_ids = np.digitize(s, edges[1:-1], right=False)
    rows = []
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        chunk = sess.loc[mask]
        risky = proxy_risky_session(chunk)
        risk_rate = float(risky.mean()) if len(risky) > 0 else float("nan")
        rows.append(
            {
                "bin": int(b + 1),
                "score_min": float(np.min(s[mask])),
                "score_max": float(np.max(s[mask])),
                "sessions": int(mask.sum()),
                "risk_rate": risk_rate,
            }
        )

    return pd.DataFrame(rows)


def topk_jaccard(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    budgets: Iterable[float],
) -> list[dict[str, float]]:
    """Top-k overlap between two score vectors (Jaccard)."""
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must align")

    n = len(scores_a)
    if n == 0:
        return [
            {"budget": float(b), "jaccard": 0.0, "overlap": 0, "topk": 0} for b in budgets
        ]

    finite_mask = np.isfinite(scores_a) & np.isfinite(scores_b)
    if finite_mask.sum() == 0:
        return [
            {"budget": float(b), "jaccard": 0.0, "overlap": 0, "topk": 0} for b in budgets
        ]

    scores_a = np.where(np.isfinite(scores_a), scores_a, -np.inf)
    scores_b = np.where(np.isfinite(scores_b), scores_b, -np.inf)

    out: list[dict[str, float]] = []
    order_a = np.argsort(scores_a)
    order_b = np.argsort(scores_b)

    for b in budgets:
        b = float(b)
        k = int(max(1, round(n * b)))
        top_a = set(order_a[-k:])
        top_b = set(order_b[-k:])
        inter = len(top_a & top_b)
        union = len(top_a | top_b)
        jaccard = float(inter / union) if union > 0 else 0.0
        out.append(
            {
                "budget": b,
                "jaccard": jaccard,
                "overlap": int(inter),
                "topk": int(k),
            }
        )

    return out
