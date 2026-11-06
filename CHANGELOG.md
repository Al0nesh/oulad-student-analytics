# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Multi-stage `Dockerfile` (pinned deps → test image with CPU PyTorch → wheel → non-root
  runtime image) and `.dockerignore`; pinned versions in `requirements/`.
- `docker/entrypoint.py`: `serve` mode runs the demo and serves the results page with a
  `/healthz` endpoint.
- `Jenkinsfile`: lint, unit tests, image build, smoke test, optional Docker Hub push, test
  server deployment and optional Minikube deployment; triggered by SCM polling.
- `jenkins/`: Jenkins LTS image with Docker CLI and kubectl, plugins list, Configuration as
  Code and Docker Compose file.
- `k8s/`: namespace, deployment (2 replicas, probes, rolling update) and NodePort service.
- `docs/jenkins-setup.md`: step-by-step local setup guide.
- `experiments/`: 53 enriched features (material types, inactivity gaps, per-assessment
  scores, activity relative to classmates, course context), six neural encoders, Optuna
  tuning of LightGBM, XGBoost and the networks, a weighted ensemble and the final
  four-split evaluation with paired bootstrap tests.
- `notebooks/`: `OULAD_LSTM_model.ipynb` (stage 1) and `OULAD_best_model.ipynb` (best model).
- Optional dependency group `experiments`.

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
