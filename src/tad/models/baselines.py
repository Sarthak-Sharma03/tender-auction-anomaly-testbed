from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


@dataclass
class RobustZModel:
    median: np.ndarray
    mad: np.ndarray

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mad = np.where(self.mad <= 1e-12, 1.0, self.mad)
        z = np.abs((x - self.median) / mad)
        return np.nanmean(z, axis=1)


@dataclass
class ModelWrapper:
    """Uniform `.score(X)` API.

    Convention: higher score = more anomalous.
    """

    name: str
    model: object

    def score(self, x: np.ndarray) -> np.ndarray:
        m = self.model
        if hasattr(m, "score"):
            return np.asarray(m.score(x), dtype=float)

        if hasattr(m, "score_samples"):
            return -np.asarray(m.score_samples(x), dtype=float)

        if hasattr(m, "decision_function"):
            return -np.asarray(m.decision_function(x), dtype=float)

        raise TypeError(f"Model {self.name} does not support scoring.")


def fit_baselines(x_fit: np.ndarray, *, random_state: int = 13) -> dict[str, ModelWrapper]:
    """Fit a small baseline suite.

    For tiny samples, we fall back to `robust_z` only. This keeps tests and examples stable.
    """
    x_fit = np.asarray(x_fit, dtype=float)
    if x_fit.ndim != 2:
        raise ValueError("x_fit must be 2D")

    n = x_fit.shape[0]
    median = np.nanmedian(x_fit, axis=0)
    mad = np.nanmedian(np.abs(x_fit - median), axis=0)
    robust = ModelWrapper("robust_z", RobustZModel(median=median, mad=mad))

    models: dict[str, ModelWrapper] = {robust.name: robust}

    if n < 10:
        return models

    iso = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "iso",
                IsolationForest(
                    n_estimators=200,
                    contamination="auto",
                    random_state=random_state,
                ),
            ),
        ]
    )
    iso.fit(x_fit)
    models["isolation_forest"] = ModelWrapper("isolation_forest", iso)

    oc = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ocsvm", OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")),
        ]
    )
    oc.fit(x_fit)
    models["one_class_svm"] = ModelWrapper("one_class_svm", oc)

    if n >= 25:
        k = int(min(35, n - 1))
        lof = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lof", LocalOutlierFactor(n_neighbors=k, novelty=True, contamination="auto")),
            ]
        )
        lof.fit(x_fit)
        models["local_outlier_factor"] = ModelWrapper("local_outlier_factor", lof)

    return models


def score_all(models: dict[str, ModelWrapper], x: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, m in models.items():
        out[name] = m.score(x)
    return out
