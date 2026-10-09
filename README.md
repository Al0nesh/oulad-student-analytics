# oulad-student-analytics

[![CI](https://github.com/Al0nesh/oulad-student-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/Al0nesh/oulad-student-analytics/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)

A small, tested Python tool for **early prediction of at-risk students** from the
[Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset).

It is the data-processing module of the master's thesis *"Application of Artificial
Intelligence Methods for Analyzing and Predicting Students' Academic Performance and
Designing Their Educational Trajectories"* (Astana IT University, 2025–2027). It builds
the **classical machine-learning baselines** that the thesis's deep temporal model
will be compared against.

## What it does

```
OULAD CSV files ──► load & validate ──► clean ──► features up to day N ──► models ──► metrics + figures
 (5 tables)          (loader.py)      (preprocess)   (features.py)        (model.py)    (viz.py)
```

1. **Loads and validates** the five OULAD tables it needs and stops with a clear message if a file or column is missing.
2. **Cleans** demographics (fixes the known `"10-20"` typo in `imd_band`, keeps a flag for missing values, encodes ordinal categories).
3. **Builds features using only data up to a cut-off day** (default: day 70, about week 10), so the prediction is made *during* the course without looking into the future:
   - VLE activity: total clicks, active days, clicks per active day, clicks before the start, clicks for every complete week;
   - assessments: submission rate, mean and weighted score, late submissions, days submitted before the deadline;
   - demographics: education, age band, deprivation index, previous attempts, credits, gender, disability.
4. **Target:** `at_risk = 1` for *Fail* or *Withdrawn*, `0` for *Pass* or *Distinction*. Students who already withdrew before the cut-off are excluded.
5. **Trains four baselines** — majority class, logistic regression, random forest, gradient boosting — on a **70 / 15 / 15** split **grouped by student** (the same student is never in two parts).
6. **Reports** accuracy, precision, recall, F1 and ROC-AUC, feature importance and three figures.

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Al0nesh/oulad-student-analytics.git
cd oulad-student-analytics
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start (no download needed)

```bash
oulad-analytics demo --out-dir results-demo
```

This generates a synthetic dataset **in exactly the OULAD file format** and runs the whole pipeline.
The synthetic numbers only show that the pipeline works — they are **not** research results.

## Running on the real OULAD

1. Download `anonymisedData.zip` from <https://analyse.kmi.open.ac.uk/open_dataset> and unzip it.
2. Run:

```bash
oulad-analytics train --data-dir "C:\path\to\OULAD_dataset" --cutoff 70 --out-dir results
```

Other commands:

```bash
oulad-analytics features --data-dir PATH --cutoff 35 --out features.csv   # only the feature table
oulad-analytics synthetic --out fake_data --n-students 1000               # only fake data
oulad-analytics --help
```

### Output files

| File | Content |
|---|---|
| `summary.md` | dataset size, at-risk share, split sizes, test-set metrics |
| `metrics.csv` | all metrics for every model on validation and test sets |
| `feature_importance.csv` / `.png` | which features the random forest relies on |
| `weekly_clicks.png` | median weekly activity of at-risk vs. successful students |
| `model_comparison.png` | F1, recall and ROC-AUC per model |

## Project structure

```
src/oulad_analytics/
    loader.py      read + validate OULAD tables
    preprocess.py  cleaning and target variable
    features.py    feature engineering with a cut-off day
    model.py       grouped split, baseline models, metrics
    viz.py         figures
    synthetic.py   fake data in OULAD format (for tests and CI)
    cli.py         command-line interface
tests/             pytest suite (31 tests, 100 % line coverage)
.github/workflows/ ci.yml (lint, test, demo, build) and release.yml (CD)
```

## Technology choices

| Need | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | standard in data science; all ML libraries available |
| Tables | pandas | OULAD is a set of relational CSV tables; joins and group-bys are one line |
| Models and metrics | scikit-learn | well-tested baselines, pipelines, grouped splits, reproducible with a seed |
| Figures | matplotlib | publication-quality static figures, works on servers without a screen |
| Tests | pytest + pytest-cov | simple test functions, fixtures, coverage gate (80 %) |
| Style | ruff | linter and formatter in one fast tool |
| CI/CD | GitHub Actions | built into GitHub, free for public repositories |

## Testing

```bash
pytest --cov          # tests + coverage
ruff check .          # lint
ruff format --check . # formatting
```

The tests use a three-student dataset where every feature value is checked by hand,
plus synthetic data for end-to-end runs. They also check that **events after the cut-off
are ignored** and that **no student appears in two splits** — the two most common
sources of data leakage in learning analytics.

## CI/CD

- **CI** (`.github/workflows/ci.yml`) — on every push and pull request: lint → tests on
  Python 3.10–3.13 (Ubuntu) and 3.12 (Windows) → demo run on synthetic data (results shown on
  the run page and saved as an artifact) → package build.
- **CD** (`.github/workflows/release.yml`) — pushing a tag such as `v0.1.0` runs the tests,
  builds the wheel, and publishes a GitHub Release with the package and demo results attached.
- **Dependabot** checks for new versions of dependencies and actions every month.

## Data and licence

The OULAD data is **not** stored in this repository. It is published by the Open University
under CC BY 4.0. If you use it, cite:

> Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset.
> *Scientific Data*, 4, 170171. https://doi.org/10.1038/sdata.2017.171

The code is released under the [MIT License](LICENSE). To cite this software, see
[`CITATION.cff`](CITATION.cff).

## Roadmap

- [ ] Run on the full OULAD and record the baseline results
- [ ] Compare several cut-off days (week 4, 8, 12, 16)
- [ ] Add an LSTM model on the weekly click sequence (thesis hypothesis)
- [ ] Add SHAP explanations for individual students
- [ ] Recommend next modules (educational trajectory) from the predictions
