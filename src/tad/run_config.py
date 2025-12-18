from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .quickstart import run_quickstart


def run_from_yaml(config_path: str | Path) -> None:
    p = Path(config_path)
    cfg: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))
    cmd = str(cfg.get("cmd", "quickstart")).strip().lower()
    if cmd != "quickstart":
        raise ValueError(f"Unsupported cmd in config: {cmd}")

    run_quickstart(
        data_path=cfg.get("data_path", "data/Login_Data.xlsx"),
        out_dir=cfg.get("out_dir", "reports/run_from_config"),
        budgets=cfg.get("budgets", [0.005, 0.01, 0.05]),
        time_split=cfg.get("time_split", 0.7),
        random_state=cfg.get("random_state", 13),
        synthetic=bool(cfg.get("synthetic", False)),
        synthetic_cfg=cfg.get("synthetic_cfg", None),
        ensemble_method=cfg.get("ensemble_method", "fisher"),
    )
