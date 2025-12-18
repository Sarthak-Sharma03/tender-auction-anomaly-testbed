from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _md_table(df: pd.DataFrame) -> str:
    """Markdown table without pandas display truncation."""
    with pd.option_context(
        "display.max_columns",
        None,
        "display.max_colwidth",
        None,
        "display.width",
        10_000,
    ):
        return df.to_markdown(index=False)


def plot_score_histogram(scores: pd.Series, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = scores.to_numpy(dtype=float)
    plt.figure()
    plt.hist(arr[~np.isnan(arr)], bins=40)
    plt.title(title)
    plt.xlabel("score (higher = more anomalous)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_budget_curve(
    budgets: list[float],
    precisions: list[float],
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(budgets, precisions, marker="o")
    plt.title(title)
    plt.xlabel("budget (fraction flagged)")
    plt.ylabel("precision@budget")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_risk_curve(risk_curve: pd.DataFrame, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if risk_curve.empty:
        return
    xs = risk_curve["bin"].to_numpy(dtype=float)
    ys = risk_curve["risk_rate"].to_numpy(dtype=float)
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel("score bin (low -> high)")
    plt.ylabel("proxy risk rate")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_markdown_report(
    out_dir: Path,
    *,
    run_name: str,
    data_note: str,
    session_stats: dict[str, Any],
    model_names: list[str],
    ensemble_name: str,
    budgets_table: pd.DataFrame,
    threshold_table: pd.DataFrame | None,
    drift_items: dict[str, Any],
    summary_stats: dict[str, Any] | None = None,
    risk_curve_table: pd.DataFrame | None = None,
    agreement_table: pd.DataFrame | None = None,
    extra_notes: list[str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# {run_name}")
    lines.append("")
    lines.append("A small reproducible run meant to be skimmed (not a benchmark).")
    lines.append("")
    lines.append("## Data")
    lines.append(f"- {data_note}")
    for k, v in session_stats.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Models")
    lines.append("- Baselines: " + ", ".join(model_names))
    lines.append(
        f"- Primary score used for ranking/reporting: **{ensemble_name}**"
    )
    lines.append("")

    if summary_stats:
        lines.append("## Summary")
        for k, v in summary_stats.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append("## Budget evaluation (ranked, proxy label)")
    lines.append(
        "We report precision/recall under a fixed *review budget* using "
        "failed-logins as a proxy risk signal."
    )
    lines.append("")
    lines.append(_md_table(budgets_table))
    lines.append("")

    if threshold_table is not None and len(threshold_table) > 0:
        lines.append("## Threshold transfer check (train -> test)")
        lines.append(
            "For each budget, we pick a score threshold on the train window "
            "and apply it to the later window. This is a quick sanity check "
            "that the ranking isn't only good in-sample."
        )
        lines.append("")
        lines.append(_md_table(threshold_table))
        lines.append("")

    if risk_curve_table is not None and len(risk_curve_table) > 0:
        lines.append("## Risk curve (proxy label)")
        lines.append(
            "Risk rate by score quantile; should trend upward if ranking is meaningful."
        )
        lines.append("")
        lines.append(_md_table(risk_curve_table))
        lines.append("")

    if agreement_table is not None and len(agreement_table) > 0:
        lines.append("## Ensemble agreement")
        lines.append(
            "Jaccard overlap between the top-k sessions of different ensemble methods."
        )
        lines.append("")
        lines.append(_md_table(agreement_table))
        lines.append("")

    lines.append("## Drift check (time split)")
    lines.append(
        "A lightweight drift summary between early and late windows "
        "(PSI + KS statistic). Drift isn't automatically 'bad' - it's a signal "
        "to re-check thresholds and monitoring."
    )
    lines.append("")
    top = sorted(
        drift_items.items(),
        key=lambda kv: float(kv[1].get("psi", 0.0)),
        reverse=True,
    )[:8]
    if top:
        lines.append("| item | psi | ks |")
        lines.append("|---|---:|---:|")
        for name, d in top:
            lines.append(
                f"| {name} | {float(d.get('psi', 0.0)):.4f} | {float(d.get('ks', 0.0)):.4f} | "
            )
        lines.append("")

    if extra_notes:
        lines.append("## Notes")
        for n in extra_notes:
            lines.append(f"- {n}")
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
