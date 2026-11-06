import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from oulad_analytics.cli import main  # noqa: E402
from oulad_analytics.deep import (  # noqa: E402
    SEQ_CHANNELS,
    LSTMConfig,
    _auc,
    _f1,
    build_sequences,
    paired_bootstrap,
    train_lstm,
)
from oulad_analytics.features import build_feature_table  # noqa: E402
from oulad_analytics.loader import load_oulad  # noqa: E402
from oulad_analytics.model import split_by_student  # noqa: E402

FAST = LSTMConfig(max_epochs=8, patience=3)


def test_sequences_match_hand_computed_values(tiny_data):
    """Cut-off day 20 -> 3 complete weeks (days 0-20); student 3 withdrew earlier."""
    table = build_feature_table(tiny_data, cutoff_day=20)
    seq = build_sequences(tiny_data, table, cutoff_day=20)
    assert seq.shape == (2, 3, len(SEQ_CHANNELS))
    assert list(table["id_student"]) == [1, 2]
    c = {name: i for i, name in enumerate(SEQ_CHANNELS)}

    s1, s2 = seq[0], seq[1]
    # VLE: student 1 clicked on days 0 and 3 (week 0) and day 8 (week 1); day -2 is pre-course
    np.testing.assert_allclose(s1[:, c["log_clicks"]], np.log1p([14, 6, 0]), rtol=1e-6)
    np.testing.assert_array_equal(s1[:, c["active_days"]], [2, 1, 0])
    # student 2's 100 clicks on day 50 are after the cut-off and must be ignored
    np.testing.assert_allclose(s2[:, c["log_clicks"]], np.log1p([3, 0, 0]), rtol=1e-6)

    # deadlines: day 7 -> week 1, day 14 -> week 2; day 100 and the exam are excluded
    for s in (s1, s2):
        np.testing.assert_array_equal(s[:, c["n_due"]], [0, 1, 1])

    # student 1: on time on day 6 (80) and day 14 (60); day-99 submission is ignored
    np.testing.assert_array_equal(s1[:, c["n_submitted"]], [1, 0, 1])
    np.testing.assert_allclose(s1[:, c["mean_score"]], [0.8, 0, 0.6], rtol=1e-6)
    np.testing.assert_array_equal(s1[:, c["n_late"]], [0, 0, 0])
    # student 2: late on day 9 (deadline 7) with 40 points
    np.testing.assert_array_equal(s2[:, c["n_submitted"]], [0, 1, 0])
    np.testing.assert_allclose(s2[:, c["mean_score"]], [0, 0.4, 0], rtol=1e-6)
    np.testing.assert_array_equal(s2[:, c["n_late"]], [0, 1, 0])


def test_sequences_reject_bad_input(tiny_data):
    table = build_feature_table(tiny_data, cutoff_day=20)
    with pytest.raises(ValueError, match="complete week"):
        build_sequences(tiny_data, table, cutoff_day=5)
    with pytest.raises(ValueError, match="one row per enrolment"):
        build_sequences(tiny_data, pd.concat([table, table]), cutoff_day=20)


@pytest.fixture(scope="module")
def synthetic(synthetic_dir):
    data = load_oulad(synthetic_dir)
    table = build_feature_table(data, cutoff_day=70)
    return data, table, split_by_student(table, seed=3)


def test_lstm_is_reproducible_and_learns(synthetic):
    data, table, split = synthetic
    r1 = train_lstm(data, table, split, cutoff_day=70, config=FAST, seed=5)
    r2 = train_lstm(data, table, split, cutoff_day=70, config=FAST, seed=5)
    pd.testing.assert_frame_equal(r1.metrics, r2.metrics)

    test = r1.metrics.set_index("split").loc["test"]
    assert test[["accuracy", "precision", "recall", "f1", "roc_auc"]].between(0, 1).all()
    assert test["roc_auc"] > 0.6  # the synthetic data has a real signal
    assert len(r1.test_score) == len(split.test)
    assert set(np.unique(r1.test_pred)) <= {0, 1}
    assert 1 <= len(r1.model.history) <= FAST.max_epochs


def test_lstm_requires_default_index(synthetic):
    data, table, split = synthetic
    with pytest.raises(ValueError, match="index"):
        train_lstm(data, table.set_index(table.index + 5), split, cutoff_day=70, config=FAST)


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 50)
    groups = np.arange(100) // 2  # two rows per student
    noise = rng.random(100)

    same = paired_bootstrap(y, groups, noise, noise, n_boot=200)
    assert same["diff"] == 0 and same["p_value"] == 1.0

    better = paired_bootstrap(y, groups, y * 0.9 + 0.05, noise, n_boot=200)
    assert better["a"] == 1.0 and better["diff"] > 0
    assert better["ci_low"] > 0 and better["p_value"] < 0.05

    f1 = paired_bootstrap(y, groups, y, 1 - y, metric="f1", n_boot=50)
    assert f1["diff"] == 1.0

    with pytest.raises(ValueError):
        paired_bootstrap(y, groups, y, y, metric="accuracy")


def test_fast_metrics_match_scikit_learn():
    from sklearn.metrics import f1_score, roc_auc_score

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    score = np.round(rng.random(300), 2)  # rounding creates ties
    pred = (score > 0.4).astype(int)
    assert _auc(y, score) == pytest.approx(roc_auc_score(y, score))
    assert _f1(y, pred) == pytest.approx(f1_score(y, pred))
    assert _auc(np.zeros(4), np.linspace(0, 1, 4)) == 0.5
    assert _f1(np.zeros(3), np.zeros(3)) == 0.0


def test_train_command_with_deep(synthetic_dir, tmp_path):
    out = tmp_path / "res"
    assert main(["train", "--data-dir", str(synthetic_dir), "--out-dir", str(out), "--deep"]) == 0
    metrics = pd.read_csv(out / "metrics.csv")
    assert {"lstm"} <= set(metrics["model"])
    sig = pd.read_csv(out / "significance.csv")
    assert set(sig["metric"]) == {"roc_auc", "f1"}
    assert "majority_class" not in set(sig["baseline"])
    assert (out / "lstm_history.csv").stat().st_size > 0
    assert "LSTM vs. baselines" in (out / "summary.md").read_text(encoding="utf-8")
