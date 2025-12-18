from __future__ import annotations

import numpy as np
import pandas as pd

from tad.eval.metrics import compute_budget_metrics


def test_budget_metrics_schema() -> None:
    sessions = pd.DataFrame({"failed_events": [0, 1, 0, 1, 1, 0]})
    scores = pd.Series([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    rows = compute_budget_metrics(sessions, scores, budgets=[0.5], model="m")
    assert len(rows) == 1
    row = rows[0]
    assert 0.0 <= row.precision <= 1.0
    assert 0.0 <= row.recall <= 1.0
    assert row.precision_ci_low <= row.precision_ci_high
    assert row.recall_ci_low <= row.recall_ci_high
    assert row.flagged_sessions > 0
