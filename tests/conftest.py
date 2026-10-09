"""Shared test fixtures.

Two kinds of data are used:

* ``tiny_data`` — a hand-made dataset with three students where every feature
  value can be checked by hand;
* ``synthetic_dir`` — a larger generated dataset in OULAD file format for
  end-to-end tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oulad_analytics.loader import OuladData
from oulad_analytics.synthetic import write_synthetic


def _info(sid: int, result: str, imd: str | None = "20-30%") -> dict:
    return {
        "code_module": "AAA",
        "code_presentation": "2013J",
        "id_student": sid,
        "gender": "M" if sid % 2 else "F",
        "region": "Wales",
        "highest_education": "A Level or Equivalent",
        "imd_band": imd,
        "age_band": "0-35",
        "num_of_prev_attempts": 0,
        "studied_credits": 60,
        "disability": "N",
        "final_result": result,
    }


@pytest.fixture
def tiny_data() -> OuladData:
    """Three students in one module.

    * student 1 — passes; 2 assessments on time, clicks in weeks 0 and 1;
    * student 2 — fails; 1 late submission, little activity, plus activity
      after the cut-off day that must be ignored;
    * student 3 — withdrew on day 10 (before the cut-off) and must be dropped.
    """
    student_info = pd.DataFrame(
        [_info(1, "Pass"), _info(2, "Fail", imd="10-20"), _info(3, "Withdrawn", imd=None)]
    )
    registration = pd.DataFrame(
        {
            "code_module": "AAA",
            "code_presentation": "2013J",
            "id_student": [1, 2, 3],
            "date_registration": [-30, -10, -5],
            "date_unregistration": [np.nan, np.nan, 10],
        }
    )
    assessments = pd.DataFrame(
        {
            "code_module": "AAA",
            "code_presentation": "2013J",
            "id_assessment": [11, 12, 13, 14],
            "assessment_type": ["TMA", "TMA", "TMA", "Exam"],
            "date": [7, 14, 100, np.nan],  # 3rd TMA is after the cut-off
            "weight": [25.0, 75.0, 0.0, 100.0],
        }
    )
    student_assessment = pd.DataFrame(
        {
            "id_assessment": [11, 12, 11, 13],
            "id_student": [1, 1, 2, 1],
            "date_submitted": [6, 14, 9, 99],
            "is_banked": 0,
            "score": [80.0, 60.0, 40.0, 90.0],
        }
    )
    student_vle = pd.DataFrame(
        {
            "code_module": "AAA",
            "code_presentation": "2013J",
            "id_student": [1, 1, 1, 1, 2, 2, 3],
            "id_site": 1,
            "date": [-2, 0, 3, 8, 1, 50, 2],  # student 2 day 50 > cut-off 20
            "sum_click": [5, 10, 4, 6, 3, 100, 7],
        }
    )
    return OuladData(student_info, registration, assessments, student_assessment, student_vle)


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    return write_synthetic(tmp_path_factory.mktemp("oulad"), n_students=300, seed=1)
