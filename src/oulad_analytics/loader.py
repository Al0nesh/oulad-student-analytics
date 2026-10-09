"""Loading and schema validation of the raw OULAD tables.

OULAD (Open University Learning Analytics Dataset, Kuzilek et al., 2017) is
distributed as seven CSV files. Only the five tables needed for the
prediction pipeline are loaded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Columns each table must contain for the pipeline to work.
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "studentInfo": [
        "code_module",
        "code_presentation",
        "id_student",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "num_of_prev_attempts",
        "studied_credits",
        "disability",
        "final_result",
    ],
    "studentRegistration": [
        "code_module",
        "code_presentation",
        "id_student",
        "date_registration",
        "date_unregistration",
    ],
    "assessments": [
        "code_module",
        "code_presentation",
        "id_assessment",
        "assessment_type",
        "date",
        "weight",
    ],
    "studentAssessment": [
        "id_assessment",
        "id_student",
        "date_submitted",
        "is_banked",
        "score",
    ],
    "studentVle": [
        "code_module",
        "code_presentation",
        "id_student",
        "id_site",
        "date",
        "sum_click",
    ],
}

KEY = ["code_module", "code_presentation", "id_student"]


class SchemaError(ValueError):
    """Raised when an OULAD table is missing or lacks required columns."""


@dataclass
class OuladData:
    """Container for the OULAD tables used by the pipeline."""

    student_info: pd.DataFrame
    registration: pd.DataFrame
    assessments: pd.DataFrame
    student_assessment: pd.DataFrame
    student_vle: pd.DataFrame

    @property
    def n_enrolments(self) -> int:
        return len(self.student_info)


def validate_columns(name: str, df: pd.DataFrame) -> None:
    """Check that ``df`` has every column required for table ``name``."""
    missing = [c for c in REQUIRED_COLUMNS[name] if c not in df.columns]
    if missing:
        raise SchemaError(f"Table '{name}' is missing columns: {', '.join(missing)}")


def _read_table(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise SchemaError(f"File not found: {path}")
    # OULAD marks missing values with '?' in a few columns.
    df = pd.read_csv(path, na_values=["?", ""], low_memory=False)
    validate_columns(name, df)
    return df


def load_oulad(data_dir: str | Path) -> OuladData:
    """Read and validate the OULAD CSV files in ``data_dir``.

    Parameters
    ----------
    data_dir:
        Folder that contains ``studentInfo.csv``, ``studentRegistration.csv``,
        ``assessments.csv``, ``studentAssessment.csv`` and ``studentVle.csv``.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise SchemaError(f"Data directory not found: {data_dir}")

    return OuladData(
        student_info=_read_table(data_dir, "studentInfo"),
        registration=_read_table(data_dir, "studentRegistration"),
        assessments=_read_table(data_dir, "assessments"),
        student_assessment=_read_table(data_dir, "studentAssessment"),
        student_vle=_read_table(data_dir, "studentVle"),
    )
