from __future__ import annotations

import numpy as np
import pandas as pd

from tad.eval.diagnostics import score_risk_curve, topk_jaccard


def test_topk_jaccard_nan_safe() -> None:
    scores_a = np.array([1.0, np.nan, 0.5, 2.0])
    scores_b = np.array([1.1, 0.2, np.nan, 2.1])
    rows = topk_jaccard(scores_a, scores_b, budgets=[0.5])
    assert len(rows) == 1
    assert rows[0]["topk"] == 2
    assert 0.0 <= rows[0]["jaccard"] <= 1.0


def test_topk_jaccard_all_nan() -> None:
    scores_a = np.array([np.nan, np.nan])
    scores_b = np.array([np.nan, np.nan])
    rows = topk_jaccard(scores_a, scores_b, budgets=[0.5])
    assert rows[0]["topk"] == 0
    assert rows[0]["jaccard"] == 0.0
    assert rows[0]["overlap"] == 0


def test_score_risk_curve_schema() -> None:
    sessions = pd.DataFrame({"failed_events": [0, 1, 0, 1]})
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    curve = score_risk_curve(sessions, scores, bins=2)
    assert set(["bin", "score_min", "score_max", "sessions", "risk_rate"]).issubset(
        curve.columns
    )
