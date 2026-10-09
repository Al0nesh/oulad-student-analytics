# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-10

### Added
- Loading and schema validation of five OULAD tables.
- Cleaning of `studentInfo` and binary `at_risk` target.
- Feature engineering with a cut-off day (VLE clicks per week, assessment scores, late submissions).
- Student-grouped 70/15/15 split and four baseline models with five metrics.
- Figures: weekly clicks by outcome, feature importance, model comparison.
- Synthetic OULAD-format data generator and `oulad-analytics` CLI.
- pytest suite, ruff configuration, GitHub Actions CI and release workflows.
