import pandas as pd
import pytest

from oulad_analytics.loader import REQUIRED_COLUMNS, SchemaError, load_oulad, validate_columns


def test_load_synthetic_dataset(synthetic_dir):
    data = load_oulad(synthetic_dir)
    assert data.n_enrolments == 300
    for name, cols in REQUIRED_COLUMNS.items():
        assert set(cols) <= set(pd.read_csv(synthetic_dir / f"{name}.csv", nrows=1).columns)


def test_question_mark_is_read_as_missing(tmp_path, synthetic_dir):
    # OULAD uses "?" for missing values; copy the data and inject one.
    for f in synthetic_dir.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    info = pd.read_csv(tmp_path / "studentInfo.csv")
    info["imd_band"] = info["imd_band"].astype(object)
    info.loc[0, "imd_band"] = "?"
    info.to_csv(tmp_path / "studentInfo.csv", index=False)

    data = load_oulad(tmp_path)
    assert pd.isna(data.student_info.loc[0, "imd_band"])


def test_missing_directory_raises(tmp_path):
    with pytest.raises(SchemaError, match="not found"):
        load_oulad(tmp_path / "nope")


def test_missing_file_raises(tmp_path):
    with pytest.raises(SchemaError, match="studentInfo.csv"):
        load_oulad(tmp_path)


def test_missing_column_raises():
    df = pd.DataFrame({"id_student": [1]})
    with pytest.raises(SchemaError, match="final_result"):
        validate_columns("studentInfo", df)
