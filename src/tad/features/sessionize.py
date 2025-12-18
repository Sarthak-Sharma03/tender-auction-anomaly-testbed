from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = [
    "Login Timestamp",
    "User ID",
    "IP Address",
    "Country",
    "Browser Name and Version",
    "Device Type",
    "Login Successful",
]


@dataclass(frozen=True)
class SessionizeConfig:
    # user-day default
    mode: str = "user_day"


def validate_events(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["Login Timestamp"] = pd.to_datetime(out["Login Timestamp"], errors="coerce")
    out = out.dropna(subset=["Login Timestamp"]).copy()
    out = out.sort_values("Login Timestamp")
    return out


def sessionize_user_day(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["date"] = tmp["Login Timestamp"].dt.date.astype(str)
    tmp["session_id"] = tmp["User ID"].astype(str) + "::" + tmp["date"]
    tmp["failed"] = (tmp["Login Successful"].astype(int) == 0).astype(int)

    def _mode(x: Iterable[object]) -> str:
        s = pd.Series(list(x))
        if len(s) == 0:
            return ""
        return str(s.mode().iloc[0])

    def _unique_hours(x: Iterable[object]) -> int:
        s = pd.Series(list(x))
        if len(s) == 0:
            return 0
        return int(pd.to_datetime(s, errors="coerce").dt.hour.nunique())

    agg = tmp.groupby("session_id", as_index=False).agg(
        user_id=("User ID", "first"),
        ip_mode=("IP Address", _mode),
        country_mode=("Country", _mode),
        events=("session_id", "size"),
        failed_events=("failed", "sum"),
        unique_ips=("IP Address", lambda x: int(pd.Series(list(x)).nunique())),
        unique_countries=("Country", lambda x: int(pd.Series(list(x)).nunique())),
        unique_devices=("Device Type", lambda x: int(pd.Series(list(x)).nunique())),
        unique_browsers=("Browser Name and Version", lambda x: int(pd.Series(list(x)).nunique())),
        unique_hours=("Login Timestamp", _unique_hours),
        first_ts=("Login Timestamp", "min"),
        last_ts=("Login Timestamp", "max"),
    )
    agg["failed_rate"] = agg["failed_events"] / agg["events"].clip(lower=1)
    agg["session_start"] = agg["first_ts"]
    agg["session_end"] = agg["last_ts"]
    return agg.sort_values("first_ts").reset_index(drop=True)


def sessionize(df: pd.DataFrame, cfg: SessionizeConfig | None = None) -> pd.DataFrame:
    cfg = cfg or SessionizeConfig()
    if cfg.mode == "user_day":
        return sessionize_user_day(df)
    raise ValueError(f"Unknown sessionize mode: {cfg.mode}")
