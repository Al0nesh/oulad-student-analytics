import pandas as pd
import pytest

from oulad_analytics.preprocess import clean_student_info, make_target


def test_target_marks_fail_and_withdrawn_as_at_risk():
    s = pd.Series(["Pass", "Distinction", "Fail", "Withdrawn"])
    assert make_target(s).tolist() == [0, 0, 1, 1]


def test_target_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unexpected"):
        make_target(pd.Series(["Pass", "Excellent"]))


def test_clean_student_info(tiny_data):
    df = clean_student_info(tiny_data.student_info)
    # the "10-20" typo is fixed and mapped to band 1; missing IMD is flagged
    assert df["imd_band"].tolist()[:2] == [2, 1]
    assert pd.isna(df["imd_band"].iloc[2])
    assert df["imd_missing"].tolist() == [0, 0, 1]
    assert df["highest_education"].unique().tolist() == [2]
    assert df["gender_m"].tolist() == [1, 0, 1]
    assert "gender" not in df.columns
    assert df["at_risk"].tolist() == [0, 1, 1]


def test_duplicates_are_removed(tiny_data):
    doubled = pd.concat([tiny_data.student_info, tiny_data.student_info])
    assert len(clean_student_info(doubled)) == 3


def test_input_is_not_modified(tiny_data):
    before = tiny_data.student_info.copy()
    clean_student_info(tiny_data.student_info)
    pd.testing.assert_frame_equal(before, tiny_data.student_info)
