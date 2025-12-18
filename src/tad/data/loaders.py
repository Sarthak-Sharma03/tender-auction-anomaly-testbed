from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_login_events(path: str | Path) -> pd.DataFrame:
    """Load login-like events from an Excel/CSV file.

    The repo ships a sample dataset at `data/Login_Data.xlsx`. It is a tender/auction surrogate
    with the public field mapping documented in `docs/DOMAIN_CONTEXT.md`. If you replace it with
    your own export, keep column names stable or map them before calling the pipeline.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    # normalize columns
    renames: dict[str, str] = {}
    for c in df.columns:
        if c.strip().lower() in {"login timestamp", "timestamp", "time"}:
            renames[c] = "Login Timestamp"
        if c.strip().lower() in {"user id", "userid", "user"}:
            renames[c] = "User ID"
        if c.strip().lower() in {"login successful", "success", "login_successful"}:
            renames[c] = "Login Successful"
    if renames:
        df = df.rename(columns=renames)

    return df
