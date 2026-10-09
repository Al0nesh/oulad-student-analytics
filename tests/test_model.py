import numpy as np
import pytest

from oulad_analytics.features import build_feature_table
from oulad_analytics.loader import load_oulad
from oulad_analytics.model import (
    evaluate,
    feature_importance,
    split_by_student,
    train_and_evaluate,
)


@pytest.fixture(scope="module")
def table(synthetic_dir):
    return build_feature_table(load_oulad(synthetic_dir), cutoff_day=70)


def test_split_sizes_and_no_student_overlap(table):
    split = split_by_student(table, seed=0)
    sizes = split.sizes()
    assert sum(sizes.values()) == len(table)
    assert 0.6 < sizes["train"] / len(table) < 0.8
    ids = [set(p["id_student"]) for p in (split.train, split.val, split.test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


@pytest.mark.parametrize("train,val", [(0, 0.1), (0.9, 0.2), (1.2, 0.1)])
def test_split_rejects_bad_fractions(table, train, val):
    with pytest.raises(ValueError):
        split_by_student(table, train=train, val=val)


def test_evaluate_perfect_prediction():
    y = np.array([0, 1, 1, 0])
    m = evaluate(y, y, y.astype(float))
    assert m == {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0}


def test_evaluate_single_class_gives_nan_auc():
    y = np.zeros(3, dtype=int)
    assert np.isnan(evaluate(y, y, np.zeros(3))["roc_auc"])


def test_training_is_reproducible_and_beats_majority(table):
    r1, _ = train_and_evaluate(table, seed=7)
    r2, _ = train_and_evaluate(table, seed=7)
    assert r1.metrics.equals(r2.metrics)

    test = r1.metrics[r1.metrics["split"] == "test"].set_index("model")
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    assert test[metric_cols].apply(lambda c: c.between(0, 1)).all().all()
    # every real model must learn something beyond guessing the majority class
    for name in ("logistic_regression", "random_forest", "gradient_boosting"):
        assert test.loc[name, "roc_auc"] > test.loc["majority_class", "roc_auc"]


def test_feature_importance(table):
    result, _ = train_and_evaluate(table, seed=0)
    imp = feature_importance(result)
    assert imp.is_monotonic_decreasing
    assert imp.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        feature_importance(result, "logistic_regression")
