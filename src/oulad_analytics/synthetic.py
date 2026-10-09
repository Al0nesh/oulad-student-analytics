"""Generate a small synthetic dataset with exactly the OULAD file layout.

The real OULAD is ~450 MB and is not stored in the repository (it is
distributed under CC BY 4.0 by the Open University). The synthetic data lets
the tests and the CI pipeline run everywhere in a few seconds. A hidden
"engagement" variable drives both the activity and the final result, so models
have a realistic signal to learn — but the numbers are **not** real results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MODULES = [("AAA", "2013J"), ("AAA", "2014J"), ("BBB", "2013J"), ("BBB", "2014B")]
COURSE_LENGTH = 240
TMA_DAYS = [19, 54, 89, 124, 159, 194]  # tutor-marked assessment deadlines


def generate(n_students: int = 400, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Return a dict of DataFrames named like the OULAD CSV files."""
    rng = np.random.default_rng(seed)

    info_rows, reg_rows, sa_rows, vle_rows = [], [], [], []
    assessments = pd.DataFrame(
        [
            {
                "code_module": m,
                "code_presentation": p,
                "id_assessment": 1000 + i * 10 + k,
                "assessment_type": "TMA",
                "date": d,
                "weight": 100 / len(TMA_DAYS),
            }
            for i, (m, p) in enumerate(MODULES)
            for k, d in enumerate(TMA_DAYS)
        ]
        + [
            {
                "code_module": m,
                "code_presentation": p,
                "id_assessment": 1000 + i * 10 + 9,
                "assessment_type": "Exam",
                "date": np.nan,
                "weight": 100.0,
            }
            for i, (m, p) in enumerate(MODULES)
        ]
    )

    for sid in range(1, n_students + 1):
        mod_idx = rng.integers(len(MODULES))
        module, pres = MODULES[mod_idx]
        engagement = rng.normal()
        prev = int(rng.poisson(0.3))
        risk_logit = -0.4 - 1.6 * engagement + 0.5 * prev + rng.normal(scale=0.6)
        at_risk = rng.random() < 1 / (1 + np.exp(-risk_logit))
        withdrawn = at_risk and rng.random() < 0.5
        result = (
            ("Withdrawn" if withdrawn else "Fail")
            if at_risk
            else ("Distinction" if engagement > 1.2 and rng.random() < 0.6 else "Pass")
        )
        unreg_day = float(rng.integers(5, 200)) if withdrawn else np.nan

        info_rows.append(
            {
                "code_module": module,
                "code_presentation": pres,
                "id_student": sid,
                "gender": rng.choice(["M", "F"]),
                "region": rng.choice(["Scotland", "London Region", "Wales", "North Region"]),
                "highest_education": rng.choice(
                    ["Lower Than A Level", "A Level or Equivalent", "HE Qualification"],
                    p=[0.4, 0.45, 0.15],
                ),
                "imd_band": rng.choice(["0-10%", "10-20", "20-30%", "50-60%", "90-100%", None]),
                "age_band": rng.choice(["0-35", "35-55", "55<="], p=[0.7, 0.28, 0.02]),
                "num_of_prev_attempts": prev,
                "studied_credits": int(rng.choice([30, 60, 90, 120])),
                "disability": rng.choice(["N", "Y"], p=[0.9, 0.1]),
                "final_result": result,
            }
        )
        reg_rows.append(
            {
                "code_module": module,
                "code_presentation": pres,
                "id_student": sid,
                "date_registration": int(rng.integers(-150, -1)),
                "date_unregistration": unreg_day,
            }
        )

        last_day = int(unreg_day) if withdrawn else COURSE_LENGTH
        daily_rate = np.exp(1.2 + 0.7 * engagement)
        for day in range(-10, last_day, 1):
            fade = 0.6 if at_risk and day > 60 else 1.0
            if rng.random() < min(0.95, 0.25 + 0.15 * engagement) * fade:
                vle_rows.append(
                    {
                        "code_module": module,
                        "code_presentation": pres,
                        "id_student": sid,
                        "id_site": int(rng.integers(500000, 500050)),
                        "date": day,
                        "sum_click": int(rng.poisson(daily_rate)) + 1,
                    }
                )

        for k, deadline in enumerate(TMA_DAYS):
            if deadline > last_day or rng.random() > 0.55 + 0.25 * engagement:
                continue  # did not submit
            delay = int(rng.integers(-6, 3) + (4 if at_risk and rng.random() < 0.3 else 0))
            score = float(np.clip(rng.normal(68 + 12 * engagement, 12), 0, 100))
            sa_rows.append(
                {
                    "id_assessment": 1000 + mod_idx * 10 + k,
                    "id_student": sid,
                    "date_submitted": deadline + delay,
                    "is_banked": 0,
                    "score": round(score),
                }
            )

    return {
        "studentInfo": pd.DataFrame(info_rows),
        "studentRegistration": pd.DataFrame(reg_rows),
        "assessments": assessments,
        "studentAssessment": pd.DataFrame(sa_rows),
        "studentVle": pd.DataFrame(vle_rows),
    }


def write_synthetic(out_dir: str | Path, n_students: int = 400, seed: int = 0) -> Path:
    """Write the synthetic tables as OULAD-named CSV files into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in generate(n_students, seed).items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return out_dir
