"""Optuna search over sequence architectures, inputs and training settings (validation ROC-AUC)."""

import json
import sys
from pathlib import Path

import optuna
from common import RESULTS, load, split
from nets import train

optuna.logging.set_verbosity(optuna.logging.WARNING)
n_trials = int(sys.argv[1])
kinds = (
    sys.argv[2].split(",")
    if len(sys.argv) > 2
    else ["lstm", "bilstm", "gru", "lstm_attn", "transformer"]
)
tag = sys.argv[3] if len(sys.argv) > 3 else "nn"

tab, seq, static, meta = load(70)
tr, va, te = split(tab)
y = tab["at_risk"].to_numpy()
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
# static inputs: demographics/course (20) or every tabular feature (base + enriched)
full_tab = tab[meta["base_features"] + meta["enriched"]].astype(float).to_numpy()
ST_SETS = {"demo20": static, "full_tab": full_tab}


def objective(t):
    kind = t.suggest_categorical("kind", kinds)
    chs = t.suggest_categorical("channels", list(CH_SETS))
    sts = t.suggest_categorical("static_set", list(ST_SETS))
    params = dict(
        hidden=t.suggest_categorical("hidden", [32, 64, 128]),
        layers=t.suggest_int("layers", 1, 2),
        dropout=t.suggest_float("dropout", 0.0, 0.5),
        lr=t.suggest_float("lr", 3e-4, 3e-3, log=True),
        weight_decay=t.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        batch=t.suggest_categorical("batch", [128, 256, 512]),
        static_hidden=t.suggest_categorical("static_hidden", [16, 32, 64]),
    )
    net, sc, auc, ep = train(kind, seq[:, :, CH_SETS[chs]], ST_SETS[sts], y, tr, va, **params)
    t.set_user_attr("epochs", ep)
    return auc


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.enqueue_trial(
    {
        "kind": kinds[0],
        "channels": "orig6",
        "static_set": "full_tab",
        "hidden": 64,
        "layers": 1,
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch": 256,
        "static_hidden": 32,
    }
)
study.optimize(
    objective,
    n_trials=n_trials,
    callbacks=[lambda s, t: print(t.number, round(t.value, 4), t.params, flush=True)],
)
best = {"val_auc": study.best_value, **study.best_params}
print("BEST", best, flush=True)
Path(RESULTS / f"best_{tag}.json").write_text(json.dumps(best, indent=1))
study.trials_dataframe().to_csv(RESULTS / f"trials_{tag}.csv", index=False)
