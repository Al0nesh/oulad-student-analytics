# Accuracy experiments (stage 2)

Scripts behind the article *Early Prediction of At-Risk Students in Online Learning:
Temporal Deep Learning, Gradient Boosting and Hybrid Ensembles on OULAD*.
They extend the package with 53 engineered features, six neural encoders, Optuna tuning,
a weighted ensemble and a paired-bootstrap comparison over four student-grouped splits.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[experiments]"
```

Folders (environment variables, defaults in brackets):

| Variable | Meaning |
|---|---|
| `OULAD_DATA` | OULAD CSV files, including `vle.csv` and `courses.csv` [`../data`] |
| `OULAD_CACHE` | cached features and sequences [`experiments/cache`] |
| `OULAD_RESULTS` | tuned parameters, metrics, scores [`experiments/results`] |

## Run order

Run from the `experiments` folder. Times are for a 2-core CPU.

| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `python prep.py $OULAD_DATA cache 70` | features and weekly sequences at day 70 | ~1 min |
| 2 | `python tab_quick.py` | base vs enriched features, validation | ~30 s |
| 3 | `python nn_quick.py` | six encoders with default settings | ~4 min |
| 4 | `python tune_tab.py lgbm 100` | Optuna for LightGBM | ~7 min |
| 5 | `python tune_nn.py 60` | Optuna over encoders, inputs and training | ~80 min |
| 6 | `python tune_tab.py xgb 60` | Optuna for XGBoost | ~9 min |
| 7 | `python final.py 42,1,7,2026` | final test evaluation and significance | ~40 min |

All selection uses the validation part of the split; the test part is scored once in step 7.

## Results (test, mean of four splits)

| Model | ROC-AUC | F1 |
|---|---|---|
| Logistic regression (32 base features) | 0.828 | 0.701 |
| Random forest (32 base features) | 0.842 | 0.717 |
| Hybrid LSTM, stage 1 | 0.851 | 0.726 |
| XGBoost, tuned (85 features) | 0.854 | 0.724 |
| LightGBM, tuned (85 features) | 0.855 | 0.726 |
| BiLSTM, tuned (6 weekly channels + 85 features) | 0.855 | 0.727 |
| **Ensemble LightGBM + XGBoost + BiLSTM** | **0.857** | **0.729** |

The ensemble beats the stage-1 LSTM in all four splits (paired bootstrap, p ≤ 0.001).
BiLSTM and LightGBM on the same features do not differ significantly in any split.

The two notebooks in `../notebooks` walk through stage 1 and the best model step by step.
