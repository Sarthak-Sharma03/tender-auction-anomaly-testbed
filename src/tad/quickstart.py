from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data.loaders import load_login_events
from .data.synthetic import SyntheticConfig, make_synthetic_login_events
from .eval.diagnostics import score_risk_curve, topk_jaccard
from .eval.drift import drift_summary
from .eval.metrics import compute_budget_metrics, threshold_eval
from .eval.report import (
    plot_budget_curve,
    plot_risk_curve,
    plot_score_histogram,
    write_markdown_report,
)
from .features.matrix import build_feature_frame, to_matrix
from .features.sessionize import sessionize, validate_events
from .models.baselines import fit_baselines, score_all
from .models.bundle import save_bundle
from .models.ensemble import fisher_ensemble, rank_ensemble


@dataclass(frozen=True)
class QuickstartConfig:
    out_dir: Path = Path("reports/quickstart")
    budgets: tuple[float, ...] = (0.005, 0.01, 0.05)
    time_split: float = 0.7
    max_fit_sessions: int = 5000
    random_state: int = 13

    # ensemble choice
    ensemble_method: str = "fisher"

    # synthetic config
    synthetic: bool = False
    synthetic_cfg: SyntheticConfig = field(default_factory=SyntheticConfig)


def _time_split_index(n: int, frac: float) -> int:
    frac = float(frac)
    frac = max(0.05, min(frac, 0.95))
    return max(1, min(int(round(n * frac)), n - 1))


def run_quickstart(
    data_path: str | Path = "data/Login_Data.xlsx",
    out_dir: str | Path = "reports/quickstart",
    budgets: Iterable[float] = (0.005, 0.01, 0.05),
    time_split: float = 0.7,
    random_state: int = 13,
    *,
    synthetic: bool = False,
    synthetic_cfg: dict[str, object] | SyntheticConfig | None = None,
    ensemble_method: str = "fisher",
) -> None:
    cfg_synth = SyntheticConfig()
    if synthetic_cfg is not None:
        if isinstance(synthetic_cfg, SyntheticConfig):
            cfg_synth = synthetic_cfg
        elif isinstance(synthetic_cfg, dict):
            try:
                cfg_synth = SyntheticConfig(**synthetic_cfg)
            except TypeError as e:
                raise ValueError(f"Invalid synthetic_cfg keys: {e}") from e
        else:
            raise TypeError("synthetic_cfg must be a dict or SyntheticConfig")

    cfg = QuickstartConfig(
        out_dir=Path(out_dir),
        budgets=tuple(float(b) for b in budgets),
        time_split=float(time_split),
        random_state=int(random_state),
        synthetic=bool(synthetic),
        synthetic_cfg=cfg_synth,
        ensemble_method=str(ensemble_method).strip().lower(),
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = cfg.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if cfg.synthetic:
        events = make_synthetic_login_events(cfg.synthetic_cfg)
        data_note = "synthetic generator (repo-contained)"
    else:
        events = load_login_events(Path(data_path))
        data_note = f"file: {Path(data_path).as_posix()}"

    validate_events(events)
    sessions = sessionize(events)
    feats = build_feature_frame(sessions)

    # stable ordering
    sessions = sessions.sort_values("session_start").reset_index(drop=True)
    feats = feats.loc[sessions.index].reset_index(drop=True)

    n = len(sessions)
    split_idx = _time_split_index(n, cfg.time_split)

    train_sessions = sessions.iloc[:split_idx].reset_index(drop=True)
    test_sessions = sessions.iloc[split_idx:].reset_index(drop=True)

    train_feats = feats.iloc[:split_idx].reset_index(drop=True)
    # time split

    # fit cap
    fit_idx = np.arange(len(train_feats))
    if len(fit_idx) > cfg.max_fit_sessions:
        rng = np.random.default_rng(cfg.random_state)
        fit_idx = rng.choice(fit_idx, size=cfg.max_fit_sessions, replace=False)
        fit_idx = np.sort(fit_idx)

    X_fit = to_matrix(train_feats.iloc[fit_idx])
    fitted = fit_baselines(X_fit, random_state=cfg.random_state)

    # score all
    X_all = to_matrix(feats)
    scores = score_all(fitted, X_all)

    baseline_names = list(fitted.keys())
    scores["ensemble_rank"] = rank_ensemble(
        {k: scores[k] for k in baseline_names},
        members=baseline_names,
    )

    # calibrated ensemble
    train_scores = {
        k: np.asarray(scores[k], dtype=float)[:split_idx] for k in baseline_names
    }
    scores["ensemble_fisher"] = fisher_ensemble(
        {k: scores[k] for k in baseline_names},
        train_scores,
        members=baseline_names,
    )

    primary = (
        "ensemble_fisher"
        if cfg.ensemble_method in {"fisher", "fisher_p", "fisher_stat"}
        else "ensemble_rank"
    )

    score_df = pd.DataFrame(
        {
            "session_id": sessions["session_id"],
            **{k: v for k, v in scores.items()},
        }
    )
    score_df.to_csv(cfg.out_dir / "scores.csv", index=False)

    # budget metrics
    budgets = list(cfg.budgets)
    all_budget_rows = []
    for name, s in scores.items():
        bm = compute_budget_metrics(sessions, pd.Series(s), budgets, model=name)
        all_budget_rows.extend([r.__dict__ for r in bm])

    (cfg.out_dir / "budget_metrics.json").write_text(
        json.dumps(all_budget_rows, indent=2),
        encoding="utf-8",
    )

    # drift summary
    drift = drift_summary(features=feats, scores=scores, split_idx=split_idx)
    (cfg.out_dir / "drift.json").write_text(
        json.dumps(drift, indent=2),
        encoding="utf-8",
    )

    # top sessions
    top = sessions.copy()
    top["score"] = scores[primary]
    top = top.sort_values("score", ascending=False)
    top_cols = [
        "session_id",
        "user_id",
        "session_start",
        "events",
        "failed_events",
        "score",
    ]
    top.head(200)[top_cols].to_csv(cfg.out_dir / "top_sessions.csv", index=False)

    # plot outputs
    plot_score_histogram(
        pd.Series(scores[primary]),
        plots_dir / "score_hist__primary.png",
        f"{primary} score histogram",
    )
    # precision curve
    ens_rows = [r for r in all_budget_rows if r["model"] == primary]
    ens_rows_sorted = sorted(ens_rows, key=lambda d: float(d["budget"]))
    plot_budget_curve(
        [float(r["budget"]) for r in ens_rows_sorted],
        [float(r["precision"]) for r in ens_rows_sorted],
        plots_dir / "budget_curve__primary.png",
        f"Precision vs budget ({primary})",
    )

    budgets_table = pd.DataFrame(ens_rows_sorted)[
        [
            "budget",
            "flagged_sessions",
            "risky_sessions_flagged",
            "risky_sessions_total",
            "precision",
            "precision_ci_low",
            "precision_ci_high",
            "recall",
            "recall_ci_low",
            "recall_ci_high",
            "lift",
            "failed_events_flagged",
        ]
    ]
    budgets_table = budgets_table.rename(
        columns={
            "flagged_sessions": "flagged",
            "risky_sessions_flagged": "risky_flagged",
            "risky_sessions_total": "risky_total",
            "failed_events_flagged": "failed_flagged",
        }
    )
    budgets_table["precision_ci"] = budgets_table.apply(
        lambda row: f"[{row['precision_ci_low']:.3f}, {row['precision_ci_high']:.3f}]",
        axis=1,
    )
    budgets_table["recall_ci"] = budgets_table.apply(
        lambda row: f"[{row['recall_ci_low']:.3f}, {row['recall_ci_high']:.3f}]",
        axis=1,
    )
    budgets_table = budgets_table.drop(
        columns=[
            "precision_ci_low",
            "precision_ci_high",
            "recall_ci_low",
            "recall_ci_high",
        ]
    )
    budgets_table["precision"] = budgets_table["precision"].map(lambda x: f"{x:.4f}")
    budgets_table["recall"] = budgets_table["recall"].map(lambda x: f"{x:.4f}")
    budgets_table["lift"] = budgets_table["lift"].map(lambda x: f"{x:.2f}")

    # threshold transfer
    threshold_rows = []
    for name, s in scores.items():
        te = threshold_eval(
            sessions_train=train_sessions,
            score_train=np.asarray(s)[:split_idx],
            sessions_test=test_sessions,
            score_test=np.asarray(s)[split_idx:],
            budgets=budgets,
            model=name,
        )
        threshold_rows.extend([r.__dict__ for r in te])

    (cfg.out_dir / "budget_threshold_eval.json").write_text(
        json.dumps(threshold_rows, indent=2),
        encoding="utf-8",
    )

    ens_thr = [r for r in threshold_rows if r["model"] == primary]
    ens_thr_sorted = sorted(ens_thr, key=lambda d: float(d["budget"]))
    threshold_table = pd.DataFrame(ens_thr_sorted)[
        [
            "budget",
            "threshold",
            "flagged_sessions_test",
            "flagged_rate_test",
            "risky_sessions_flagged_test",
            "risky_sessions_total_test",
            "precision_test",
            "recall_test",
            "lift_test",
            "failed_events_flagged_test",
        ]
    ]
    threshold_table = threshold_table.rename(
        columns={
            "flagged_sessions_test": "flagged_test",
            "flagged_rate_test": "flagged_rate",
            "risky_sessions_flagged_test": "risky_flagged_test",
            "risky_sessions_total_test": "risky_total_test",
            "failed_events_flagged_test": "failed_flagged_test",
        }
    )
    threshold_table["flagged_rate"] = threshold_table["flagged_rate"].map(lambda x: f"{x:.4f}")
    threshold_table["precision_test"] = threshold_table["precision_test"].map(
        lambda x: f"{x:.4f}"
    )
    threshold_table["recall_test"] = threshold_table["recall_test"].map(lambda x: f"{x:.4f}")
    threshold_table["lift_test"] = threshold_table["lift_test"].map(lambda x: f"{x:.2f}")

    session_stats = {
        "events": int(len(events)),
        "sessions": int(n),
        "train_sessions": int(len(train_sessions)),
        "test_sessions": int(len(test_sessions)),
        "time_split": cfg.time_split,
        "random_state": cfg.random_state,
        "max_fit_sessions": cfg.max_fit_sessions,
    }

    model_names = baseline_names
    extra_notes = [
        "failed_events is used as a proxy label for a quick decision-metric sanity check; "
        "treat it as a signal, not ground truth.",
        "quickstart caps the fit window for speed; "
        "see docs/EXPERIMENTS.md for config-driven runs.",
        "precision/recall intervals use Wilson binomial bounds.",
    ]

    if ens_rows_sorted:
        base_rate = float(ens_rows_sorted[0]["base_rate"])
        auc_budget = float(
            np.trapz(
                [float(r["precision"]) for r in ens_rows_sorted],
                [float(r["budget"]) for r in ens_rows_sorted],
            )
        )
    else:
        base_rate = float("nan")
        auc_budget = float("nan")
    summary_stats = {
        "base_risk_rate": f"{base_rate:.4f}",
        "auc_precision_budget": f"{auc_budget:.4f}",
    }

    risk_curve = score_risk_curve(sessions, scores[primary], bins=10)
    risk_curve.to_json(cfg.out_dir / "risk_curve.json", orient="records", indent=2)
    plot_risk_curve(
        risk_curve,
        plots_dir / "risk_curve__primary.png",
        f"Risk curve ({primary})",
    )
    risk_curve_table = risk_curve.copy()
    if not risk_curve_table.empty:
        risk_curve_table["score_min"] = risk_curve_table["score_min"].map(
            lambda x: f"{x:.3f}"
        )
        risk_curve_table["score_max"] = risk_curve_table["score_max"].map(
            lambda x: f"{x:.3f}"
        )
        risk_curve_table["risk_rate"] = risk_curve_table["risk_rate"].map(
            lambda x: f"{x:.3f}"
        )

    agreement_rows = []
    if "ensemble_fisher" in scores and "ensemble_rank" in scores:
        agreement_rows = topk_jaccard(
            scores["ensemble_fisher"],
            scores["ensemble_rank"],
            budgets,
        )
    (cfg.out_dir / "ensemble_agreement.json").write_text(
        json.dumps(agreement_rows, indent=2),
        encoding="utf-8",
    )

    agreement_table = pd.DataFrame(agreement_rows)
    if not agreement_table.empty:
        agreement_table["jaccard"] = agreement_table["jaccard"].map(lambda x: f"{x:.3f}")
        agreement_table["overlap"] = agreement_table["overlap"].astype(int)
        agreement_table["topk"] = agreement_table["topk"].astype(int)

    write_markdown_report(
        cfg.out_dir,
        run_name="Quickstart run",
        data_note=data_note,
        session_stats=session_stats,
        model_names=model_names,
        ensemble_name=primary,
        budgets_table=budgets_table,
        threshold_table=threshold_table,
        drift_items=drift,
        summary_stats=summary_stats,
        risk_curve_table=risk_curve_table,
        agreement_table=agreement_table,
        extra_notes=extra_notes,
    )

    # bundle write
    bundle_meta = {
        "created_by": "tad quickstart",
        "version": "0.7.0",
        "budgets": budgets,
        "time_split": cfg.time_split,
        "feature_columns": list(feats.columns),
        "primary_ensemble": primary,
        "ensemble_method": cfg.ensemble_method,
        "ensemble_members": baseline_names,
    }

    bundle_dir = cfg.out_dir / "model_bundle"
    save_bundle(
        bundle_dir,
        metadata=bundle_meta,
        objects=fitted,
    )

    # calibration sketch
    probs = np.linspace(0.0, 1.0, 101)
    qmap: dict[str, list[float]] = {}
    for name, t in train_scores.items():
        tt = np.asarray(t, dtype=float)
        tt = tt[~np.isnan(tt)]
        if tt.size == 0:
            qmap[name] = [0.0 for _ in probs]
            continue
        q = np.quantile(tt, probs, method="linear")
        qmap[name] = [float(x) for x in q]

    calib = {
        "probs": [float(p) for p in probs],
        "quantiles": qmap,
    }
    (bundle_dir / "calibration.json").write_text(
        json.dumps(calib, indent=2),
        encoding="utf-8",
    )
