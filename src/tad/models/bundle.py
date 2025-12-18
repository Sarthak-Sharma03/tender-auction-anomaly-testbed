from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def save_bundle(
    bundle_dir: str | Path,
    *,
    metadata: dict[str, Any],
    objects: dict[str, Any],
) -> Path:
    """Save a minimal, inspectable model bundle.

    This is deliberately plain: a reviewer can open metadata.json and see exactly what ran.
    """
    d = Path(bundle_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for name, obj in objects.items():
        joblib.dump(obj, d / f"{name}.joblib")

    return d
