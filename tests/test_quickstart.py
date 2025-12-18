from __future__ import annotations

from pathlib import Path

import pandas as pd

from tad.quickstart import run_quickstart


def test_quickstart_smoke(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "Login Timestamp": pd.date_range("2025-01-01", periods=20, freq="h"),
            "User ID": ["u1"] * 10 + ["u2"] * 10,
            "IP Address": ["1.1.1.1"] * 5 + ["2.2.2.2"] * 5 + ["3.3.3.3"] * 10,
            "Country": ["IN"] * 20,
            "Region": ["X"] * 20,
            "City": ["Y"] * 20,
            "Browser Name and Version": ["Chrome 1"] * 20,
            "Device Type": ["Desktop"] * 20,
            "Login Successful": [1] * 18 + [0, 0],
        }
    )
    data_path = tmp_path / "sample.xlsx"
    df.to_excel(data_path, index=False)

    out_dir = tmp_path / "out"
    run_quickstart(data_path=data_path, out_dir=out_dir, budgets=[0.1, 0.2], time_split=0.5)

    assert (out_dir / "report.md").exists()
    assert (out_dir / "top_sessions.csv").exists()
    assert (out_dir / "budget_metrics.json").exists()
    assert (out_dir / "drift.json").exists()
    assert (out_dir / "risk_curve.json").exists()
    assert (out_dir / "ensemble_agreement.json").exists()
