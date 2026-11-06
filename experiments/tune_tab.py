"""Optuna tuning of LightGBM / XGBoost on enriched features; objective = validation ROC-AUC."""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import optuna
import xgboost as xgb
from common import RESULTS, load, split
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
which, n_trials = sys.argv[1], int(sys.argv[2])
tab, seq, static, meta = load(70)
tr, va, te = split(tab)
y = tab["at_risk"].to_numpy()
cols = meta["base_features"] + meta["enriched"]
X = tab[cols].astype(float).to_numpy()


def lgb_obj(t):
    p = dict(
        n_estimators=3000,
        learning_rate=t.suggest_float("lr", 0.01, 0.1, log=True),
        num_leaves=t.suggest_int("num_leaves", 8, 128, log=True),
        min_child_samples=t.suggest_int("min_child_samples", 10, 200, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30, log=True),
        reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
        verbose=-1,
        random_state=42,
        n_jobs=2,
    )
    m = lgb.LGBMClassifier(**p)
    m.fit(
        X[tr],
        y[tr],
        eval_set=[(X[va], y[va])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    t.set_user_attr("n_estimators", int(m.best_iteration_))
    return roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])


def xgb_obj(t):
    p = dict(
        n_estimators=3000,
        learning_rate=t.suggest_float("lr", 0.01, 0.1, log=True),
        max_depth=t.suggest_int("max_depth", 3, 9),
        min_child_weight=t.suggest_float("min_child_weight", 0.5, 30, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30, log=True),
        gamma=t.suggest_float("gamma", 1e-3, 5, log=True),
        tree_method="hist",
        eval_metric="auc",
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=2,
    )
    m = xgb.XGBClassifier(**p)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    t.set_user_attr("n_estimators", int(m.best_iteration) + 1)
    return roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lgb_obj if which == "lgbm" else xgb_obj, n_trials=n_trials)
best = {"val_auc": study.best_value, **study.best_params, **study.best_trial.user_attrs}
print(which, best, flush=True)
Path(RESULTS / f"best_{which}.json").write_text(json.dumps(best, indent=1))
study.trials_dataframe().to_csv(RESULTS / f"trials_{which}.csv", index=False)
