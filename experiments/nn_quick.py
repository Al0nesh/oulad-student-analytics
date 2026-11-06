"""Default-settings comparison of architectures on VALIDATION ROC-AUC."""

import sys
import time

import pandas as pd
from common import RESULTS, load, split
from nets import train

tab, seq, static, meta = load(70)
tr, va, te = split(tab)
y = tab["at_risk"].to_numpy()
ch = meta["channels"]
orig6 = [
    ch.index(c)
    for c in ["log_clicks", "active_days", "n_due", "n_submitted", "mean_score", "n_late"]
]
# original static set of the first LSTM: 8 demographics + clicks_pre (first 8 + index of clicks_pre)
st_names = meta["static"]
orig_static = [
    st_names.index(c)
    for c in [
        "gender_m",
        "disability_y",
        "highest_education",
        "age_band",
        "imd_band",
        "imd_missing",
        "num_of_prev_attempts",
        "studied_credits",
        "clicks_pre",
    ]
]
runs = [("lstm", "orig 6ch / orig static", seq[:, :, orig6], static[:, orig_static])]
kinds = (
    sys.argv[1].split(",")
    if len(sys.argv) > 1
    else ["lstm", "gru", "bilstm", "lstm_attn", "cnn", "transformer"]
)
runs += [(k, "enriched 26ch / 20 static", seq, static) for k in kinds]
rows = []
for kind, data_name, s, x in runs:
    t0 = time.time()
    net, sc, auc, ep = train(kind, s, x, y, tr, va)
    rows.append(
        {
            "model": kind,
            "data": data_name,
            "val_auc": auc,
            "epochs": ep,
            "sec": round(time.time() - t0),
        }
    )
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RESULTS / "nn_quick.csv", index=False)
