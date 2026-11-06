"""Shared helpers for the accuracy experiments."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from oulad_analytics.model import split_by_student

# Folders can be changed with environment variables; defaults are next to this script.
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("OULAD_DATA", HERE.parent / "data"))
CACHE = Path(os.environ.get("OULAD_CACHE", HERE / "cache"))
RESULTS = Path(os.environ.get("OULAD_RESULTS", HERE / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)


def load(cutoff=70):
    d = CACHE / f"cutoff_{cutoff}"
    tab = pd.read_pickle(d / "tab.pkl")
    seq = np.load(d / "seq.npy")
    static = np.load(d / "static.npy")
    meta = json.loads(Path(d / "meta.json").read_text())
    return tab, seq, static, meta


def split(tab, seed=42):
    """Same student-grouped 70/15/15 split as the package (positional indices)."""
    s = split_by_student(tab, seed=seed)
    return s.train.index.to_numpy(), s.val.index.to_numpy(), s.test.index.to_numpy()


def best_threshold(y, score):
    """Threshold that maximises F1 on the given (validation) data."""
    cands = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 97)))
    f1s = [f1_score(y, score >= t) for t in cands]
    return float(cands[int(np.argmax(f1s))])


def metrics(y, score, thr=0.5):
    pred = (score >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score),
        "threshold": thr,
    }
