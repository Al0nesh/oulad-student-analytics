"""Quick check on VALIDATION only: do enriched features help tabular models?"""

import lightgbm as lgb
import pandas as pd
from common import RESULTS, load, metrics, split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

tab, seq, static, meta = load(70)
tr, va, te = split(tab)
y = tab["at_risk"].to_numpy()
sets = {
    "base (32)": meta["base_features"],
    "base+enriched": meta["base_features"] + meta["enriched"],
}
rows = []
for sname, cols in sets.items():
    X = tab[cols].astype(float)
    models = {
        "logreg": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        ),
        "rf": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                300, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=42
            ),
        ),
        "lgbm_default": lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            verbose=-1,
            random_state=42,
        ),
    }
    for mname, m in models.items():
        m.fit(X.iloc[tr], y[tr])
        r = metrics(y[va], m.predict_proba(X.iloc[va])[:, 1])
        rows.append({"features": sname, "model": mname, "val_auc": r["roc_auc"]})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RESULTS / "tab_quick.csv", index=False)
