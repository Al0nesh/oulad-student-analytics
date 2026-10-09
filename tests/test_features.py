import pytest

from oulad_analytics.features import build_feature_table, feature_columns
from oulad_analytics.loader import load_oulad

CUTOFF = 20


@pytest.fixture
def table(tiny_data):
    return build_feature_table(tiny_data, cutoff_day=CUTOFF).set_index("id_student")


def test_withdrawn_before_cutoff_is_dropped(table):
    assert sorted(table.index) == [1, 2]


def test_vle_features_by_hand(table):
    s1 = table.loc[1]
    assert s1["total_clicks"] == 5 + 10 + 4 + 6
    assert s1["active_days"] == 4
    assert s1["clicks_pre"] == 5
    assert (s1["clicks_w0"], s1["clicks_w1"], s1["clicks_w2"]) == (14, 6, 0)


def test_events_after_cutoff_are_ignored(table):
    # student 2 clicked 100 times on day 50, after the cut-off
    assert table.loc[2, "total_clicks"] == 3
    # student 1 submitted assessment 13 on day 99, after the cut-off
    assert table.loc[1, "n_submitted"] == 2


def test_assessment_features_by_hand(table):
    s1, s2 = table.loc[1], table.loc[2]
    assert s1["n_due"] == 2  # TMA 11 and 12; exam and late TMA excluded
    assert s1["mean_score"] == pytest.approx(70.0)
    assert s1["weighted_score"] == pytest.approx((80 * 25 + 60 * 75) / 100)
    assert s1["submission_rate"] == 1.0
    assert s1["late_rate"] == 0.0
    assert s2["n_late"] == 1 and s2["late_rate"] == 1.0
    assert s2["submission_rate"] == 0.5


def test_no_activity_gives_zeros(tiny_data):
    tiny_data.student_vle = tiny_data.student_vle.iloc[0:0]
    tiny_data.student_assessment = tiny_data.student_assessment.iloc[0:0]
    t = build_feature_table(tiny_data, cutoff_day=CUTOFF)
    assert (t["total_clicks"] == 0).all()
    assert (t["no_submissions"] == 1).all()
    assert (t["mean_score"] == 0).all()


def test_negative_cutoff_rejected(tiny_data):
    with pytest.raises(ValueError):
        build_feature_table(tiny_data, cutoff_day=-1)


def test_feature_columns_exclude_target_and_ids(table):
    cols = feature_columns(table.reset_index())
    assert "at_risk" not in cols and "id_student" not in cols and "final_result" not in cols
    assert "total_clicks" in cols


def test_synthetic_table_has_no_missing_except_imd(synthetic_dir):
    t = build_feature_table(load_oulad(synthetic_dir), cutoff_day=70)
    na = t[feature_columns(t)].isna().sum()
    assert set(na[na > 0].index) <= {"imd_band"}
    assert t["at_risk"].isin([0, 1]).all()
