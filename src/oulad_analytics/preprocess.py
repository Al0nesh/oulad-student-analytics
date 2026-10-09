"""Cleaning of demographic data and construction of the target variable."""

from __future__ import annotations

import pandas as pd

# final_result values in OULAD.
AT_RISK_RESULTS = {"Fail", "Withdrawn"}
VALID_RESULTS = {"Pass", "Distinction", "Fail", "Withdrawn"}

# Ordinal encodings: the order carries meaning, so these are mapped to integers
# instead of being one-hot encoded.
EDUCATION_ORDER = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4,
}
AGE_ORDER = {"0-35": 0, "35-55": 1, "55<=": 2}
IMD_ORDER = {
    "0-10%": 0,
    "10-20%": 1,
    "20-30%": 2,
    "30-40%": 3,
    "40-50%": 4,
    "50-60%": 5,
    "60-70%": 6,
    "70-80%": 7,
    "80-90%": 8,
    "90-100%": 9,
}


def make_target(final_result: pd.Series) -> pd.Series:
    """Binary target: 1 = at risk (Fail or Withdrawn), 0 = Pass or Distinction."""
    unknown = set(final_result.dropna().unique()) - VALID_RESULTS
    if unknown:
        raise ValueError(f"Unexpected final_result values: {sorted(unknown)}")
    return final_result.isin(AT_RISK_RESULTS).astype(int).rename("at_risk")


def clean_student_info(student_info: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of ``studentInfo`` with numeric demographic columns.

    * fixes the known ``"10-20"`` typo in ``imd_band`` (missing ``%``);
    * maps ordinal categories to integers and keeps missing ``imd_band`` as NaN
      plus an explicit ``imd_missing`` flag (missingness can itself be informative);
    * one-hot encodes ``gender`` and ``disability`` as 0/1 columns;
    * removes duplicate enrolment rows.
    """
    df = student_info.copy()
    df = df.drop_duplicates(subset=["code_module", "code_presentation", "id_student"])

    imd = df["imd_band"].replace({"10-20": "10-20%"})
    df["imd_missing"] = imd.isna().astype(int)
    df["imd_band"] = imd.map(IMD_ORDER)

    df["highest_education"] = df["highest_education"].map(EDUCATION_ORDER)
    df["age_band"] = df["age_band"].map(AGE_ORDER)
    df["gender_m"] = (df["gender"] == "M").astype(int)
    df["disability_y"] = (df["disability"] == "Y").astype(int)

    df["at_risk"] = make_target(df["final_result"])
    return df.drop(columns=["gender", "disability"])
