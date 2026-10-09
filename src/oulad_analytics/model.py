"""Baseline classifiers for at-risk prediction and their evaluation.

These are the *traditional machine-learning baselines* that the deep temporal
model of the thesis will later be compared against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import feature_columns


@dataclass
class Split:
    """Train / validation / test partitions of a feature table."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


@dataclass
class TrainResult:
    """Fitted models and their metrics."""

    models: dict[str, Pipeline]
    metrics: pd.DataFrame
    features: list[str] = field(default_factory=list)


def split_by_student(
    table: pd.DataFrame, train: float = 0.70, val: float = 0.15, seed: int = 42
) -> Split:
    """Split into train/val/test (default 70/15/15) **grouped by student**.

    A student can be enrolled in several modules. Grouping guarantees that the
    same student never appears in two partitions, which would leak information.
    """
    if not 0 < train < 1 or not 0 < val < 1 or train + val >= 1:
        raise ValueError("train and val must be in (0, 1) and sum to less than 1")

    groups = table["id_student"]
    outer = GroupShuffleSplit(n_splits=1, train_size=train, random_state=seed)
    train_idx, rest_idx = next(outer.split(table, groups=groups))
    rest = table.iloc[rest_idx]

    val_share = val / (1 - train)
    inner = GroupShuffleSplit(n_splits=1, train_size=val_share, random_state=seed)
    val_idx, test_idx = next(inner.split(rest, groups=rest["id_student"]))
    return Split(table.iloc[train_idx], rest.iloc[val_idx], rest.iloc[test_idx])


def make_models(seed: int = 42) -> dict[str, Pipeline]:
    """The set of baseline models. Missing values are imputed with the median."""

    def pipe(clf: ClassifierMixin, scale: bool = False) -> Pipeline:
        steps = [SimpleImputer(strategy="median")]
        if scale:
            steps.append(StandardScaler())
        return make_pipeline(*steps, clf)

    return {
        "majority_class": pipe(DummyClassifier(strategy="most_frequent")),
        "logistic_regression": pipe(
            LogisticRegression(max_iter=2000, class_weight="balanced"), scale=True
        ),
        "random_forest": pipe(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            )
        ),
        "gradient_boosting": pipe(HistGradientBoostingClassifier(random_state=seed)),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Classification metrics used in the thesis plan (positive class = at risk)."""
    has_both = len(np.unique(y_true)) == 2
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if has_both else float("nan"),
    }


def train_and_evaluate(table: pd.DataFrame, seed: int = 42) -> tuple[TrainResult, Split]:
    """Fit every baseline on the train split and score it on validation and test."""
    cols = feature_columns(table)
    split = split_by_student(table, seed=seed)
    x_train, y_train = split.train[cols], split.train["at_risk"]

    rows, fitted = [], {}
    for name, model in make_models(seed).items():
        model.fit(x_train, y_train)
        fitted[name] = model
        for part_name, part in (("val", split.val), ("test", split.test)):
            y_score = model.predict_proba(part[cols])[:, 1]
            y_pred = model.predict(part[cols])
            rows.append(
                {
                    "model": name,
                    "split": part_name,
                    **evaluate(part["at_risk"].to_numpy(), y_pred, y_score),
                }
            )

    metrics = pd.DataFrame(rows)
    return TrainResult(models=fitted, metrics=metrics, features=cols), split


def feature_importance(result: TrainResult, model_name: str = "random_forest") -> pd.Series:
    """Impurity-based importance of each feature for a tree model, sorted descending."""
    clf = result.models[model_name][-1]
    if not hasattr(clf, "feature_importances_"):
        raise ValueError(f"Model '{model_name}' does not expose feature importances")
    return pd.Series(clf.feature_importances_, index=result.features).sort_values(ascending=False)
