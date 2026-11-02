# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-02

### Added
- `deep` module: weekly multi-source sequences (VLE activity + assessment timeline) built
  only from complete weeks before the cut-off day.
- Hybrid LSTM model (`TemporalRiskNet`): LSTM over the weekly sequence + dense layer over
  demographics; class-weighted loss, early stopping on validation ROC-AUC, fixed seed.
- Paired bootstrap test that resamples students and reports the LSTM-minus-baseline
  difference in ROC-AUC and F1 with a 95 % confidence interval and p-value.
- `--deep` flag for `train` and `demo`; new outputs `significance.csv` and `lstm_history.csv`.
- Optional dependency group `deep` (PyTorch); CI installs CPU-only PyTorch.

## [0.1.0] - 2026-10-10

### Added
- Loading and schema validation of five OULAD tables.
- Cleaning of `studentInfo` and binary `at_risk` target.
- Feature engineering with a cut-off day (VLE clicks per week, assessment scores, late submissions).
- Student-grouped 70/15/15 split and four baseline models with five metrics.
- Figures: weekly clicks by outcome, feature importance, model comparison.
- Synthetic OULAD-format data generator and `oulad-analytics` CLI.
- pytest suite, ruff configuration, GitHub Actions CI and release workflows.
