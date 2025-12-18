from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    # numeric only
    include_duration: bool = True
    include_rates: bool = True


def build_feature_frame(sessions: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    spec = spec or FeatureSpec()

    f = pd.DataFrame(index=sessions.index)
    f["events"] = sessions["events"].astype(float)
    f["failed_events"] = sessions["failed_events"].astype(float)
    f["failed_rate"] = sessions["failed_rate"].astype(float)
    f["unique_ips"] = sessions["unique_ips"].astype(float)
    f["unique_countries"] = sessions["unique_countries"].astype(float)
    f["unique_devices"] = sessions["unique_devices"].astype(float)
    f["unique_browsers"] = sessions["unique_browsers"].astype(float)
    f["unique_hours"] = sessions["unique_hours"].astype(float)

    if spec.include_duration:
        dur = (sessions["last_ts"] - sessions["first_ts"]).dt.total_seconds()
        f["duration_sec"] = dur.fillna(0.0).astype(float)
        dur_hours = dur.clip(lower=60.0) / 3600.0
        f["events_per_hour"] = (sessions["events"].astype(float) / dur_hours).astype(float)
        f["failed_per_hour"] = (sessions["failed_events"].astype(float) / dur_hours).astype(float)

    if spec.include_rates:
        denom = sessions["events"].clip(lower=1).astype(float)
        f["unique_ip_rate"] = (sessions["unique_ips"].astype(float) / denom).astype(float)
        f["unique_country_rate"] = (
            sessions["unique_countries"].astype(float) / denom
        ).astype(float)
        f["unique_device_rate"] = (sessions["unique_devices"].astype(float) / denom).astype(float)
        f["unique_browser_rate"] = (sessions["unique_browsers"].astype(float) / denom).astype(float)
        f["unique_hour_rate"] = (sessions["unique_hours"].astype(float) / denom).astype(float)

    # sanitize cols
    f = f.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return f


def to_matrix(features: pd.DataFrame) -> np.ndarray:
    return features.to_numpy(dtype=float, copy=True)
