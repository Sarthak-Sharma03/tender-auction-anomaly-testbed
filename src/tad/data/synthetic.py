from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    n_users: int = 250
    days: int = 30
    mean_events_per_user_per_day: float = 4.0
    anomaly_rate: float = 0.015
    drift_day: int = 18
    random_state: int = 13


def make_synthetic_login_events(cfg: SyntheticConfig | None = None) -> pd.DataFrame:
    """Generate a synthetic login-like event table with mild concept drift.

    This exists for reviewers: it makes the repo runnable without private data.
    The goal is not perfect realism; it is a stable harness for the evaluation tooling.
    """
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.random_state)
    start = datetime(2025, 1, 1, 9, 0, 0)

    user_ids = [f"u{idx:04d}" for idx in range(cfg.n_users)]
    countries = np.array(["IN", "DE", "AT", "US", "CA", "GB"], dtype=object)
    devices = np.array(["Desktop", "Mobile", "Tablet"], dtype=object)
    browsers = np.array(["Chrome", "Edge", "Firefox", "Safari"], dtype=object)

    rows = []
    for d in range(cfg.days):
        day_ts = start + timedelta(days=d)
        drift_mult = 1.0 if d < cfg.drift_day else 1.35  # later window is noisier/busier

        for u in user_ids:
            n = rng.poisson(cfg.mean_events_per_user_per_day * drift_mult)
            for _ in range(n):
                minute = int(rng.integers(0, 24 * 60))
                ts = day_ts + timedelta(minutes=minute)

                country = str(rng.choice(countries))
                device = str(rng.choice(devices, p=[0.65, 0.30, 0.05]))
                browser = str(rng.choice(browsers, p=[0.62, 0.18, 0.12, 0.08]))
                ip_parts = (
                    int(rng.integers(1, 255)),
                    int(rng.integers(0, 255)),
                    int(rng.integers(0, 255)),
                    int(rng.integers(0, 255)),
                )
                ip = ".".join(str(p) for p in ip_parts)

                # base failure
                base_fail = 0.03 if d < cfg.drift_day else 0.05
                # anomaly mode
                is_anom = rng.random() < cfg.anomaly_rate

                fail_p = base_fail
                if is_anom:
                    fail_p = 0.75 if device != "Desktop" else 0.55
                    country = "RU" if rng.random() < 0.5 else "CN"

                success = 1 if rng.random() > fail_p else 0

                rows.append(
                    {
                        "Login Timestamp": ts,
                        "User ID": u,
                        "IP Address": ip,
                        "Country": country,
                        "Browser Name and Version": browser,
                        "Device Type": device,
                        "Login Successful": success,
                    }
                )

    df = pd.DataFrame(rows)
    df = df.sort_values("Login Timestamp").reset_index(drop=True)
    return df
