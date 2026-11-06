"""Final evaluation: every model trained on train, early-stopped / weighted / thresholded on
validation, scored ONCE on test. Repeated for several student-grouped splits.

Models
  rf_base, lstm_base   — the previous best models (package code, 32 features / 6 channels)
  lgbm, xgb            — tuned gradient boosting on enriched features (5 seeds averaged)
  nn                   — tuned neural network (5 seeds averaged)
  ensemble             — weighted average of lgbm, xgb and nn; weights chosen on validation
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from common import DATA_DIR, RESULTS, best_threshold, load, metrics, split
from nets import predict_on, train
from sklearn.metrics import roc_auc_score

from oulad_analytics.deep import STATIC_FEATURES, build_sequences, paired_bootstrap, train_lstm
from oulad_analytics.features import build_feature_table
from oulad_analytics.loader import load_oulad
from oulad_analytics.model import train_and_evaluate

SPLIT_SEEDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [42, 1, 7, 2026]
N_SEEDS = 5

tab, seq, static, meta = load(70)
y = tab["at_risk"].to_numpy()
X = tab[meta["base_features"] + meta["enriched"]].astype(float).to_numpy()
ch = meta["channels"]
CH_SETS = {
    "orig6": [
        ch.index(c)
        for c in ["log_clicks", "active_days", "n_due", "n_submitted", "mean_score", "n_late"]
    ],
    "orig6+rel": [
        ch.index(c)
        for c in [
            "log_clicks",
            "active_days",
            "n_due",
            "n_submitted",
            "mean_score",
            "n_late",
            "rel_log_clicks",
        ]
    ],
    "all26": list(range(len(ch))),
}
ST_SETS = {"demo20": static, "full_tab": X}
best_lgb = json.loads(Path(RESULTS / "best_lgbm.json").read_text())
best_xgb = json.loads(Path(RESULTS / "best_xgb.json").read_text())
best_nn = json.loads(Path(RESULTS / "best_nn.json").read_text())

data = load_oulad(DATA_DIR)
table = build_feature_table(data, cutoff_day=70)
assert (table["id_student"].to_numpy() == tab["id_student"].to_numpy()).all()
full_seq = build_sequences(data, table, 70)
st = table[STATIC_FEATURES].to_numpy(dtype=float)


def fit_lgb(tr, va, seed):
    p = dict(
        n_estimators=best_lgb["n_estimators"],
        learning_rate=best_lgb["lr"],
        num_leaves=best_lgb["num_leaves"],
        min_child_samples=best_lgb["min_child_samples"],
        subsample=best_lgb["subsample"],
        subsample_freq=1,
        colsample_bytree=best_lgb["colsample_bytree"],
        reg_lambda=best_lgb["reg_lambda"],
        reg_alpha=best_lgb["reg_alpha"],
        verbose=-1,
        random_state=seed,
        n_jobs=2,
    )
    return lgb.LGBMClassifier(**p).fit(X[tr], y[tr])


def fit_xgb(tr, va, seed):
    p = dict(
        n_estimators=best_xgb["n_estimators"],
        learning_rate=best_xgb["lr"],
        max_depth=best_xgb["max_depth"],
        min_child_weight=best_xgb["min_child_weight"],
        subsample=best_xgb["subsample"],
        colsample_bytree=best_xgb["colsample_bytree"],
        reg_lambda=best_xgb["reg_lambda"],
        gamma=best_xgb["gamma"],
        tree_method="hist",
        random_state=seed,
        n_jobs=2,
    )
    return xgb.XGBClassifier(**p).fit(X[tr], y[tr])


all_rows, sig_rows, weights_log = [], [], []
for split_seed in SPLIT_SEEDS:
    tr, va, te = split(tab, seed=split_seed)
    val_score, test_score = {}, {}

    # previous best models (exact package code)
    res, sp = train_and_evaluate(table, seed=split_seed)
    cols = res.features
    for name in ("random_forest", "logistic_regression"):
        m = res.models[name]
        val_score[name] = m.predict_proba(sp.val[cols])[:, 1]
        test_score[name] = m.predict_proba(sp.test[cols])[:, 1]
    deep = train_lstm(data, table, sp, cutoff_day=70, seed=split_seed)
    # validation predictions of the package LSTM (needed for the threshold)
    val_score["lstm_base"] = deep.model.predict_proba(full_seq[va], st[va])
    test_score["lstm_base"] = deep.test_score
    assert np.array_equal(sp.test.index.to_numpy(), te)

    # tuned boosting, averaged over seeds
    for name, fit in (("lgbm", fit_lgb), ("xgb", fit_xgb)):
        ms = [fit(tr, va, s) for s in range(N_SEEDS)]
        val_score[name] = np.mean([m.predict_proba(X[va])[:, 1] for m in ms], axis=0)
        test_score[name] = np.mean([m.predict_proba(X[te])[:, 1] for m in ms], axis=0)

    # tuned neural network, averaged over seeds
    s_in = seq[:, :, CH_SETS[best_nn["channels"]]]
    x_in = ST_SETS[best_nn["static_set"]]
    hp = {
        k: best_nn[k]
        for k in ("hidden", "layers", "dropout", "lr", "weight_decay", "batch", "static_hidden")
    }
    vs, ts = [], []
    for s in range(N_SEEDS):
        net, sc, _, _ = train(best_nn["kind"], s_in, x_in, y, tr, va, seed=s, **hp)
        vs.append(predict_on(net, sc, s_in, x_in, va))
        ts.append(predict_on(net, sc, s_in, x_in, te))
    val_score["nn"], test_score["nn"] = np.mean(vs, axis=0), np.mean(ts, axis=0)

    # ensemble weights on validation (grid, step 0.1) — rank-free probability average
    best_w, best_auc = None, -1
    for a in np.arange(0, 1.01, 0.1):
        for b in np.arange(0, 1.01 - a, 0.1):
            c = 1 - a - b
            auc = roc_auc_score(
                y[va], a * val_score["lgbm"] + b * val_score["xgb"] + c * val_score["nn"]
            )
            if auc > best_auc:
                best_auc, best_w = auc, (round(a, 1), round(b, 1), round(c, 1))
    a, b, c = best_w
    weights_log.append({"split_seed": split_seed, "w_lgbm": a, "w_xgb": b, "w_nn": c})
    val_score["ensemble"] = a * val_score["lgbm"] + b * val_score["xgb"] + c * val_score["nn"]
    test_score["ensemble"] = a * test_score["lgbm"] + b * test_score["xgb"] + c * test_score["nn"]

    for name in val_score:
        thr = best_threshold(y[va], val_score[name])
        all_rows.append(
            {
                "split_seed": split_seed,
                "model": name,
                "val_auc": roc_auc_score(y[va], val_score[name]),
                **metrics(y[te], test_score[name], thr),
            }
        )
        print(all_rows[-1], flush=True)

    groups = tab["id_student"].to_numpy()[te]
    for best in ("ensemble", "nn", "lgbm"):
        for other in ("random_forest", "lstm_base", "lgbm", "xgb", "nn"):
            if other == best:
                continue
            r = paired_bootstrap(
                y[te], groups, test_score[best], test_score[other], n_boot=2000, seed=split_seed
            )
            sig_rows.append({"split_seed": split_seed, "model": best, "vs": other, **r})
    np.savez(
        RESULTS / f"scores_{split_seed}.npz",
        te=te,
        va=va,
        **{f"test_{k}": v for k, v in test_score.items()},
        **{f"val_{k}": v for k, v in val_score.items()},
    )

pd.DataFrame(all_rows).round(4).to_csv(RESULTS / "final_metrics.csv", index=False)
pd.DataFrame(sig_rows).round(4).to_csv(RESULTS / "final_significance.csv", index=False)
pd.DataFrame(weights_log).to_csv(RESULTS / "final_weights.csv", index=False)
print("DONE", flush=True)
