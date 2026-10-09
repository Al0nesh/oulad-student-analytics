"""Feature engineering for early prediction.

All features are computed **only from events that happened on or before a
cut-off day** (``cutoff_day``, counted from the module start as in OULAD).
This simulates the real situation of a teacher who wants a prediction in the
middle of the semester and prevents information from the future leaking into
the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import KEY, OuladData
from .preprocess import clean_student_info

DEMOGRAPHIC_FEATURES = [
    "gender_m",
    "disability_y",
    "highest_education",
    "age_band",
    "imd_band",
    "imd_missing",
    "num_of_prev_attempts",
    "studied_credits",
]


def vle_features(student_vle: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """Aggregate VLE click-stream data per enrolment up to ``cutoff_day``.

    Returns total clicks, number of active days, clicks per active day,
    clicks before the course start and clicks for every complete week
    (``clicks_w0`` ... ``clicks_wN``).
    """
    vle = student_vle[student_vle["date"] <= cutoff_day]
    # Only complete weeks get their own column; a partial last week would look like a
    # sudden drop in activity. Its clicks are still counted in total_clicks.
    n_weeks = (cutoff_day + 1) // 7
    week_cols = [f"clicks_w{w}" for w in range(n_weeks)]

    if vle.empty:
        return pd.DataFrame(
            columns=KEY
            + ["total_clicks", "active_days", "clicks_per_active_day", "clicks_pre"]
            + week_cols
        )

    grouped = vle.groupby(KEY)
    out = pd.DataFrame(
        {
            "total_clicks": grouped["sum_click"].sum(),
            "active_days": grouped["date"].nunique(),
            "clicks_pre": vle[vle["date"] < 0].groupby(KEY)["sum_click"].sum(),
        }
    )
    out["clicks_pre"] = out["clicks_pre"].fillna(0)
    out["clicks_per_active_day"] = out["total_clicks"] / out["active_days"]

    started = vle[vle["date"] >= 0].assign(week=lambda d: d["date"] // 7)
    weekly = (
        started.pivot_table(index=KEY, columns="week", values="sum_click", aggfunc="sum")
        .reindex(columns=range(n_weeks))
        .fillna(0)
    )
    weekly.columns = week_cols
    out = out.join(weekly, how="left").fillna({c: 0 for c in week_cols})
    return out.reset_index()


def assessment_features(
    assessments: pd.DataFrame, student_assessment: pd.DataFrame, cutoff_day: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate assessment results per enrolment for assessments due by ``cutoff_day``.

    Final exams are excluded because they happen at the end of the course.
    Only submissions made on or before the cut-off are used.

    Returns a tuple ``(per_student, n_due_per_presentation)``.
    """
    due = assessments[
        (assessments["assessment_type"] != "Exam") & (assessments["date"] <= cutoff_day)
    ]
    n_due = due.groupby(["code_module", "code_presentation"]).size().rename("n_due").reset_index()

    subs = student_assessment[student_assessment["date_submitted"] <= cutoff_day].merge(
        due, on="id_assessment", how="inner"
    )
    if subs.empty:
        return pd.DataFrame(
            columns=KEY
            + ["n_submitted", "mean_score", "weighted_score", "n_late", "mean_days_early"]
        ), n_due

    subs = subs.assign(
        is_late=(subs["date_submitted"] > subs["date"]).astype(int),
        days_early=subs["date"] - subs["date_submitted"],
        score_x_weight=subs["score"] * subs["weight"],
    )
    g = subs.groupby(KEY)
    out = pd.DataFrame(
        {
            "n_submitted": g.size(),
            "mean_score": g["score"].mean(),
            "n_late": g["is_late"].sum(),
            "mean_days_early": g["days_early"].mean(),
            "_sw": g["score_x_weight"].sum(),
            "_w": g["weight"].sum(),
        }
    )
    # Weighted score; falls back to the mean score when all weights are 0 (e.g. CMA = 0%).
    out["weighted_score"] = np.where(
        out["_w"] > 0, out["_sw"] / out["_w"].replace(0, np.nan), out["mean_score"]
    )
    return out.drop(columns=["_sw", "_w"]).reset_index(), n_due


def build_feature_table(data: OuladData, cutoff_day: int = 70) -> pd.DataFrame:
    """Build one row per enrolment with features and the ``at_risk`` target.

    Students who had already unregistered on or before ``cutoff_day`` are
    removed: their outcome is already known, so predicting it is pointless and
    would make the task artificially easy.
    """
    if cutoff_day < 0:
        raise ValueError("cutoff_day must be >= 0")

    info = clean_student_info(data.student_info)

    reg = data.registration[KEY + ["date_registration", "date_unregistration"]]
    info = info.merge(reg, on=KEY, how="left")
    still_enrolled = info["date_unregistration"].isna() | (info["date_unregistration"] > cutoff_day)
    info = info[still_enrolled].drop(columns=["date_unregistration"])

    vle = vle_features(data.student_vle, cutoff_day)
    ass, n_due = assessment_features(data.assessments, data.student_assessment, cutoff_day)

    table = info.merge(vle, on=KEY, how="left").merge(ass, on=KEY, how="left")
    table = table.merge(n_due, on=["code_module", "code_presentation"], how="left")

    count_cols = [c for c in table.columns if c.startswith("clicks_w")] + [
        "total_clicks",
        "active_days",
        "clicks_per_active_day",
        "clicks_pre",
        "n_submitted",
        "n_late",
        "n_due",
    ]
    score_cols = ["mean_score", "weighted_score", "mean_days_early"]
    # Empty intermediate tables have object dtype; force numbers before arithmetic.
    table[count_cols + score_cols] = (
        table[count_cols + score_cols].apply(pd.to_numeric).astype(float)
    )
    table[count_cols] = table[count_cols].fillna(0)
    table["submission_rate"] = _safe_ratio(table["n_submitted"], table["n_due"])
    table["late_rate"] = _safe_ratio(table["n_late"], table["n_submitted"])
    # No submissions yet -> score 0 and "never early"; also keep an explicit flag.
    table["no_submissions"] = (table["n_submitted"] == 0).astype(int)
    table[score_cols] = table[score_cols].fillna(0)
    return table.reset_index(drop=True)


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, with 0 where the denominator is 0."""
    return (num / den.where(den > 0)).fillna(0.0)


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Names of the model input columns in a table from :func:`build_feature_table`."""
    excluded = set(KEY) | {"region", "final_result", "at_risk"}
    return [c for c in table.columns if c not in excluded]
