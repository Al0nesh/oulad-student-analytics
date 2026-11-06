"""Build enriched data for one cut-off day and cache it.

Outputs (in out_dir/cutoff_<day>/):
  tab.parquet   one row per enrolment: original 32 features + enriched features + target + keys
  seq.npy       (n, weeks, channels) weekly sequences
  static.npy    (n, s) static features for neural nets
  meta.json     channel / feature names
Uses only events on or before the cut-off day.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from oulad_analytics.features import build_feature_table, feature_columns
from oulad_analytics.loader import KEY, load_oulad

data_dir, out_root, cutoff = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
out = out_root / f"cutoff_{cutoff}"
out.mkdir(parents=True, exist_ok=True)

data = load_oulad(data_dir)
table = build_feature_table(data, cutoff_day=cutoff)  # identical rows/order to the package
base_features = feature_columns(table)
n_weeks = (cutoff + 1) // 7
rows = table[KEY].assign(_row=np.arange(len(table)))
PRES = ["code_module", "code_presentation"]

# ------------------------------------------------------------------ VLE with types
sites = pd.read_csv(data_dir / "vle.csv")[["id_site", "activity_type"]]
vle = data.student_vle
vle = vle[vle["date"] <= cutoff].merge(sites, on="id_site", how="left")
vle["activity_type"] = vle["activity_type"].fillna("other")
TYPES = sorted(vle["activity_type"].unique())

# per-type totals (for tabular models)
by_type = vle.pivot_table(
    index=KEY, columns="activity_type", values="sum_click", aggfunc="sum", fill_value=0
)
by_type.columns = [f"clk_{t}" for t in by_type.columns]
n_sites = vle.groupby(KEY)["id_site"].nunique().rename("n_distinct_sites")
by_type = by_type.join(n_sites).reset_index()

# weekly matrices (complete weeks, day >= 0)
vw = vle[(vle["date"] >= 0) & (vle["date"] < n_weeks * 7)].assign(week=lambda d: d["date"] // 7)
vw = vw.merge(rows, on=KEY, how="inner")
n = len(table)
week_clicks = np.zeros((n, n_weeks))
week_days = np.zeros((n, n_weeks))
type_week = np.zeros((n, n_weeks, len(TYPES)))
g = vw.groupby(["_row", "week"]).agg(c=("sum_click", "sum"), d=("date", "nunique")).reset_index()
week_clicks[g["_row"], g["week"]] = g["c"]
week_days[g["_row"], g["week"]] = g["d"]
gt = vw.groupby(["_row", "week", "activity_type"])["sum_click"].sum().reset_index()
gt["t"] = gt["activity_type"].map({t: i for i, t in enumerate(TYPES)})
type_week[gt["_row"], gt["week"], gt["t"]] = gt["sum_click"]

# daily active flag for streak features
vd = vle[(vle["date"] >= 0) & (vle["date"] <= cutoff)].merge(rows, on=KEY, how="inner")
daily = np.zeros((n, cutoff + 1), dtype=bool)
daily[vd["_row"], vd["date"]] = True


def longest_zero_run(a):
    best = cur = 0
    for v in a:
        cur = 0 if v else cur + 1
        best = max(best, cur)
    return best


def trailing_zero_run(a):
    k = 0
    for v in a[::-1]:
        if v:
            break
        k += 1
    return k


# ------------------------------------------------------------- assessments detail
ass = data.assessments
due = ass[(ass["assessment_type"] != "Exam") & (ass["date"] <= cutoff)].copy()
due = due.sort_values(PRES + ["date", "id_assessment"])
due["order"] = due.groupby(PRES).cumcount()
subs = data.student_assessment[data.student_assessment["date_submitted"] <= cutoff].merge(
    due, on="id_assessment", how="inner"
)
subs = subs.merge(rows, on=KEY, how="inner")
max_k = 4
first_scores = np.full((n, max_k), np.nan)
first_delay = np.full((n, max_k), np.nan)
s = subs[subs["order"] < max_k]
first_scores[s["_row"], s["order"]] = s["score"]
first_delay[s["_row"], s["order"]] = s["date_submitted"] - s["date"]
# score trend: last score minus first score among submitted
sc = subs.sort_values("date").groupby("_row")["score"]
score_trend = np.full(n, np.nan)
st = sc.last() - sc.first()
score_trend[st.index] = st.values
tma_cma = subs.pivot_table(index="_row", columns="assessment_type", values="score", aggfunc="mean")
tma_mean = np.full(n, np.nan)
cma_mean = np.full(n, np.nan)
if "TMA" in tma_cma:
    tma_mean[tma_cma.index] = tma_cma["TMA"].values
if "CMA" in tma_cma:
    cma_mean[tma_cma.index] = tma_cma["CMA"].values

# ------------------------------------------------------------------- tabular frame
tab = table.copy()
tab = tab.merge(by_type, on=KEY, how="left")
type_cols = [f"clk_{t}" for t in TYPES]
for c in type_cols + ["n_distinct_sites"]:
    if c not in tab:
        tab[c] = 0.0
tab[type_cols + ["n_distinct_sites"]] = tab[type_cols + ["n_distinct_sites"]].fillna(0)

weeks_idx = np.arange(n_weeks)
logw = np.log1p(week_clicks)
slope = ((logw - logw.mean(1, keepdims=True)) * (weeks_idx - weeks_idx.mean())).sum(1) / (
    (weeks_idx - weeks_idx.mean()) ** 2
).sum()
k = min(3, n_weeks)
tab["click_slope"] = slope
tab["last3_vs_first3"] = np.log1p(week_clicks[:, -k:].sum(1)) - np.log1p(week_clicks[:, :k].sum(1))
tab["weeks_inactive"] = (week_clicks == 0).sum(1)
tab["longest_gap_days"] = [longest_zero_run(r) for r in daily]
tab["days_since_last_active"] = [trailing_zero_run(r) for r in daily]
tab["active_days_last14"] = daily[:, -14:].sum(1)
for j in range(max_k):
    tab[f"score_a{j + 1}"] = first_scores[:, j]
    tab[f"delay_a{j + 1}"] = first_delay[:, j]
tab["score_trend"] = score_trend
tab["tma_mean"] = tma_mean
tab["cma_mean"] = cma_mean

# relative to classmates in the same module presentation (known at prediction time)
for c in ["total_clicks", "active_days", "mean_score", "weighted_score", "submission_rate"]:
    grp = tab.groupby(PRES)[c]
    tab[f"rel_{c}"] = (tab[c] - grp.transform("mean")) / (grp.transform("std") + 1e-9)
tab["rel_rank_clicks"] = tab.groupby(PRES)["total_clicks"].rank(pct=True)

# course identity
for m in sorted(tab["code_module"].unique()):
    tab[f"mod_{m}"] = (tab["code_module"] == m).astype(int)
tab["pres_B"] = tab["code_presentation"].str.endswith("B").astype(int)
tab["pres_2014"] = tab["code_presentation"].str.startswith("2014").astype(int)
courses = pd.read_csv(data_dir / "courses.csv")
tab = tab.merge(courses, on=PRES, how="left")

enriched = [
    c
    for c in tab.columns
    if c not in base_features and c not in set(KEY) | {"region", "final_result", "at_risk"}
]

# ---------------------------------------------------------------- sequences
# channels: log clicks per type, active days, total log clicks, clicks relative to course-week mean,
#           assessment timeline (due, submitted, score, late)
from oulad_analytics.deep import SEQ_CHANNELS, build_sequences  # noqa: E402

base_seq = build_sequences(data, table, cutoff)  # (n, w, 6)
course_codes = tab.groupby(PRES).ngroup().to_numpy()
rel = np.zeros((n, n_weeks))
for cidx in np.unique(course_codes):
    m = course_codes == cidx
    mu = logw[m].mean(0)
    sd = logw[m].std(0) + 1e-6
    rel[m] = (logw[m] - mu) / sd
seq = np.concatenate(
    [
        np.log1p(type_week),  # 20 type channels
        base_seq,  # 6 original channels
        rel[..., None],  # engagement relative to classmates
    ],
    axis=2,
).astype(np.float32)
CHANNELS = [f"log_{t}" for t in TYPES] + SEQ_CHANNELS + ["rel_log_clicks"]

STATIC = [
    "gender_m",
    "disability_y",
    "highest_education",
    "age_band",
    "imd_band",
    "imd_missing",
    "num_of_prev_attempts",
    "studied_credits",
    "date_registration",
    "clicks_pre",
    "module_presentation_length",
    "pres_B",
    "pres_2014",
] + [c for c in tab.columns if c.startswith("mod_")]
static = tab[STATIC].to_numpy(dtype=float)

tab.to_pickle(out / "tab.pkl")
np.save(out / "seq.npy", seq)
np.save(out / "static.npy", static)
meta = {
    "base_features": base_features,
    "enriched": enriched,
    "channels": CHANNELS,
    "static": STATIC,
    "types": TYPES,
    "n_weeks": n_weeks,
}
(out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
print(out, tab.shape, seq.shape, static.shape, "enriched:", len(enriched))
