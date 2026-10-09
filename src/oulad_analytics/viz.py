"""Figures for reports: weekly activity by outcome, feature importance, metric comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (CI servers have no screen)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RISK_COLOR = "#c0392b"
OK_COLOR = "#2471a3"


def plot_weekly_clicks(table: pd.DataFrame, path: str | Path) -> Path:
    """Median weekly VLE clicks for at-risk vs. successful students."""
    week_cols = sorted(
        (c for c in table.columns if c.startswith("clicks_w")), key=lambda c: int(c[8:])
    )
    weeks = [int(c[8:]) for c in week_cols]
    medians = table.groupby("at_risk")[week_cols].median()

    fig, ax = plt.subplots(figsize=(7, 3.8))
    if 0 in medians.index:
        ax.plot(weeks, medians.loc[0], marker="o", color=OK_COLOR, label="Pass / Distinction")
    if 1 in medians.index:
        ax.plot(weeks, medians.loc[1], marker="o", color=RISK_COLOR, label="Fail / Withdrawn")
    ax.set_xlabel("Week of the course")
    ax.set_ylabel("Median clicks in the VLE")
    ax.set_title("Weekly online activity by final outcome")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_feature_importance(importance: pd.Series, path: str | Path, top: int = 12) -> Path:
    """Horizontal bar chart of the ``top`` most important features."""
    imp = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values, color=OK_COLOR)
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.set_title("Most important features (random forest)")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def plot_metrics(metrics: pd.DataFrame, path: str | Path, split: str = "test") -> Path:
    """Grouped bar chart comparing models on F1, recall and ROC-AUC."""
    m = metrics[metrics["split"] == split].set_index("model")[["f1", "recall", "roc_auc"]]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    m.plot.bar(ax=ax, rot=0, color=["#2471a3", "#7fb3d5", "#1e8449"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title(f"Baseline models on the {split} set")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.tick_params(axis="x", labelsize=8)
    return _save(fig, path)


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
