from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from ..features.matrix import build_feature_frame, to_matrix
from ..features.sessionize import sessionize, validate_events
from ..models.ensemble import fisher_stat, rank_ensemble
from .schemas import ScoreItem, ScoreRequest, ScoreResponse


def _load_bundle(bundle_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}

    meta_path = bundle_dir / "metadata.json"
    if meta_path.exists():
        out["_metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))

    calib_path = bundle_dir / "calibration.json"
    if calib_path.exists():
        out["_calibration"] = json.loads(calib_path.read_text(encoding="utf-8"))

    for p in bundle_dir.glob("*.joblib"):
        out[p.stem] = joblib.load(p)

    return out


def _pvalues_from_quantile_cdf(
    *,
    scores: np.ndarray,
    probs: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    """Approximate one-sided p-values p = P_train(S >= s) using a quantile-CDF sketch."""
    s = np.asarray(scores, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    pgrid = np.asarray(probs, dtype=float)

    if q.size == 0 or pgrid.size == 0 or q.size != pgrid.size:
        return np.ones_like(s, dtype=float)

    # quantile safe
    cdf = np.interp(s, q, pgrid, left=0.0, right=1.0)
    p = 1.0 - cdf
    p = np.clip(p, 1e-300, 1.0)
    return np.where(np.isnan(s), 1.0, p)


def create_app(bundle_dir: str | Path) -> FastAPI:
    bundle_path = Path(bundle_dir)
    bundle = _load_bundle(bundle_path)

    app = FastAPI(title="Tender/Auction Anomaly Testbed (reference scorer)")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "bundle": str(bundle_path), "meta": bundle.get("_metadata", {})}

    @app.post("/score_sessions", response_model=ScoreResponse)
    def score_sessions(req: ScoreRequest) -> ScoreResponse:
        try:
            events = pd.DataFrame(req.events)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid events: {e}") from e

        validate_events(events)
        sessions = sessionize(events)
        feats = build_feature_frame(sessions)
        X = to_matrix(feats)

        per_model: dict[str, np.ndarray] = {}
        for name in ["robust_z", "isolation_forest", "one_class_svm", "local_outlier_factor"]:
            m = bundle.get(name)
            if m is None:
                continue
            per_model[name] = np.asarray(m.score(X), dtype=float)

        if not per_model:
            raise HTTPException(status_code=500, detail="No models found in the bundle.")

        meta = bundle.get("_metadata", {}) or {}
        primary = str(meta.get("primary_ensemble", "ensemble_rank"))

        ens = None
        if primary == "ensemble_fisher":
            calib = bundle.get("_calibration") or {}
            probs = np.asarray(calib.get("probs", []), dtype=float)
            qmap = calib.get("quantiles", {}) or {}

            pcols: list[np.ndarray] = []
            for name, s in per_model.items():
                q = np.asarray(qmap.get(name, []), dtype=float)
                pcols.append(_pvalues_from_quantile_cdf(scores=s, probs=probs, quantiles=q))

            if pcols:
                ens = fisher_stat(np.vstack(pcols))

        if ens is None:
            ens = rank_ensemble(per_model)

        sessions = sessions.copy()
        sessions["ensemble_score"] = ens

        if bool(req.return_per_model):
            for name, s in per_model.items():
                sessions[f"score__{name}"] = s

        top = sessions.sort_values("ensemble_score", ascending=False).head(int(req.top_k))

        items: list[ScoreItem] = []
        for _, row in top.iterrows():
            meta_out: dict[str, Any] = {
                "events": int(row["events"]),
                "failed_events": int(row["failed_events"]),
            }
            if bool(req.return_per_model):
                for name in per_model.keys():
                    k = f"score__{name}"
                    if k in row:
                        meta_out[k] = float(row[k])

            items.append(
                ScoreItem(
                    session_id=str(row["session_id"]),
                    score=float(row["ensemble_score"]),
                    meta=meta_out,
                )
            )

        return ScoreResponse(top=items)

    return app
