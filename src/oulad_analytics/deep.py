"""Deep temporal model of the thesis: an LSTM over weekly multi-source sequences.

The classical baselines in :mod:`oulad_analytics.model` see each student as one
row of *aggregated* numbers (total clicks, mean score, ...). The hypothesis of
the thesis is that keeping the **order of weeks** and combining **several data
sources** in one model predicts better. This module tests that hypothesis:

* :func:`build_sequences` turns the raw tables into one sequence per enrolment,
  ``(weeks, channels)``, using only complete weeks up to the cut-off day.
  The channels combine two sources: VLE activity and the assessment timeline.
* :class:`TemporalRiskNet` reads the sequence with an LSTM and the demographic
  features with a small dense layer, then joins both for the prediction.
* :func:`train_lstm` trains it on exactly the same student-grouped split as the
  baselines, so the comparison is fair.
* :func:`compare_to_baselines` checks whether the difference is statistically
  significant with a paired bootstrap that resamples **students**.

PyTorch is an optional dependency: ``pip install -e ".[deep]"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch import nn

from .features import DEMOGRAPHIC_FEATURES
from .loader import KEY, OuladData
from .model import Split, evaluate

#: What the model sees for every week (one value per channel).
SEQ_CHANNELS = [
    "log_clicks",  # log(1 + VLE clicks in the week)
    "active_days",  # days with at least one click, 0..7
    "n_due",  # non-exam assessments due this week (same for the whole module run)
    "n_submitted",  # assessments the student submitted this week
    "mean_score",  # mean score of those submissions / 100 (0 if none)
    "n_late",  # submissions made after their deadline
]

#: Non-temporal inputs: demographics plus activity before the course started.
STATIC_FEATURES = [*DEMOGRAPHIC_FEATURES, "clicks_pre"]


# --------------------------------------------------------------------------- data


def build_sequences(data: OuladData, table: pd.DataFrame, cutoff_day: int) -> np.ndarray:
    """Weekly sequences aligned row by row with ``table``.

    Parameters
    ----------
    data:
        The raw OULAD tables.
    table:
        Output of :func:`~oulad_analytics.features.build_feature_table` for the
        same ``cutoff_day``; row ``i`` of the result belongs to row ``i`` of it.
    cutoff_day:
        Last day that may be used. Only complete weeks ``0 .. (cutoff_day+1)//7 - 1``
        are kept, as in the aggregated features.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(table), n_weeks, len(SEQ_CHANNELS))``, dtype float32.
    """
    n_weeks = (cutoff_day + 1) // 7
    if n_weeks < 1:
        raise ValueError("cutoff_day must cover at least one complete week (>= 6)")
    if table[KEY].duplicated().any():
        raise ValueError("table must have one row per enrolment")

    seq = np.zeros((len(table), n_weeks, len(SEQ_CHANNELS)), dtype=np.float32)
    rows = table[KEY].assign(_row=np.arange(len(table)))
    ch = {name: i for i, name in enumerate(SEQ_CHANNELS)}

    # ---- source 1: VLE click stream (complete weeks only)
    vle = data.student_vle
    vle = vle[(vle["date"] >= 0) & (vle["date"] < n_weeks * 7)]
    vle = vle.assign(week=vle["date"] // 7)
    weekly = vle.groupby([*KEY, "week"]).agg(clicks=("sum_click", "sum"), days=("date", "nunique"))
    weekly = weekly.reset_index().merge(rows, on=KEY, how="inner")
    _put(seq, weekly, ch["log_clicks"], np.log1p(weekly["clicks"]))
    _put(seq, weekly, ch["active_days"], weekly["days"])

    # ---- source 2: assessment timeline (no final exam, deadline by the cut-off)
    ass = data.assessments
    due = ass[(ass["assessment_type"] != "Exam") & (ass["date"] <= cutoff_day)]
    due = due.assign(week=_week(due["date"], n_weeks))
    n_due = due.groupby(["code_module", "code_presentation", "week"]).size().rename("n")
    n_due = n_due.reset_index().merge(rows, on=["code_module", "code_presentation"])
    _put(seq, n_due, ch["n_due"], n_due["n"])

    subs = data.student_assessment
    subs = subs[subs["date_submitted"] <= cutoff_day].merge(
        due.drop(columns="week"), on="id_assessment", how="inner"
    )
    if not subs.empty:
        subs = subs.assign(
            week=_week(subs["date_submitted"], n_weeks),
            late=(subs["date_submitted"] > subs["date"]).astype(int),
        )
        per_week = subs.groupby([*KEY, "week"]).agg(
            n=("id_assessment", "size"), score=("score", "mean"), late=("late", "sum")
        )
        per_week = per_week.reset_index().merge(rows, on=KEY, how="inner")
        _put(seq, per_week, ch["n_submitted"], per_week["n"])
        _put(seq, per_week, ch["mean_score"], per_week["score"].fillna(0) / 100)
        _put(seq, per_week, ch["n_late"], per_week["late"])
    return seq


def _week(day: pd.Series, n_weeks: int) -> pd.Series:
    """Week number of ``day``; days before the start go to week 0, the partial
    last week (still before the cut-off) goes to the last complete week."""
    return (day // 7).clip(0, n_weeks - 1).astype(int)


def _put(seq: np.ndarray, frame: pd.DataFrame, channel: int, values: pd.Series) -> None:
    seq[frame["_row"].to_numpy(), frame["week"].to_numpy(), channel] = values.to_numpy()


# -------------------------------------------------------------------------- model


class TemporalRiskNet(nn.Module):
    """LSTM over the weekly sequence + dense layer over static features.

    ``sequence (B, T, C) -> LSTM -> last hidden state (B, H)``
    ``static (B, S) -> Linear + ReLU -> (B, Hs)``
    ``[H ; Hs] -> Dropout -> Linear + ReLU -> Linear -> logit of "at risk"``
    """

    def __init__(
        self,
        n_channels: int,
        n_static: int,
        hidden: int = 32,
        static_hidden: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True)
        self.static = nn.Sequential(nn.Linear(n_static, static_hidden), nn.ReLU())
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + static_hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(seq)
        joined = torch.cat([h[-1], self.static(static)], dim=1)
        return self.head(joined).squeeze(1)


@dataclass
class LSTMConfig:
    """Training settings. The defaults are deliberately small and stable."""

    hidden: int = 32
    static_hidden: int = 16
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 60
    patience: int = 8  # stop after this many epochs without a better validation ROC-AUC


@dataclass
class LSTMRiskModel:
    """A trained :class:`TemporalRiskNet` together with its input scaling."""

    net: TemporalRiskNet
    seq_mean: np.ndarray
    seq_std: np.ndarray
    static_median: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    history: list[dict[str, float]] = field(default_factory=list)

    def _tensors(self, seq: np.ndarray, static: np.ndarray) -> tuple[torch.Tensor, ...]:
        s = (seq - self.seq_mean) / self.seq_std
        x = np.where(np.isnan(static), self.static_median, static)
        x = (x - self.static_mean) / self.static_std
        return torch.from_numpy(s.astype(np.float32)), torch.from_numpy(x.astype(np.float32))

    def predict_proba(self, seq: np.ndarray, static: np.ndarray) -> np.ndarray:
        """Probability of being at risk for every row."""
        self.net.eval()
        s, x = self._tensors(seq, static)
        with torch.no_grad():
            return torch.sigmoid(self.net(s, x)).numpy()


def fit_lstm(
    seq_train: np.ndarray,
    static_train: np.ndarray,
    y_train: np.ndarray,
    seq_val: np.ndarray,
    static_val: np.ndarray,
    y_val: np.ndarray,
    config: LSTMConfig | None = None,
    seed: int = 42,
) -> LSTMRiskModel:
    """Train with Adam and early stopping on the validation ROC-AUC.

    Scaling statistics are computed on the training part only. The loss is
    weighted by the class ratio (like ``class_weight="balanced"`` in the
    baselines), so a 0.5 threshold is meaningful for an imbalanced target.
    """
    cfg = config or LSTMConfig()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    flat = seq_train.reshape(-1, seq_train.shape[-1])
    median = np.nanmedian(static_train, axis=0)
    median = np.where(np.isnan(median), 0.0, median)
    filled = np.where(np.isnan(static_train), median, static_train)
    model = LSTMRiskModel(
        net=TemporalRiskNet(
            seq_train.shape[-1], static_train.shape[1], cfg.hidden, cfg.static_hidden, cfg.dropout
        ),
        seq_mean=flat.mean(axis=0),
        seq_std=flat.std(axis=0) + 1e-6,
        static_median=median,
        static_mean=filled.mean(axis=0),
        static_std=filled.std(axis=0) + 1e-6,
    )

    s_tr, x_tr = model._tensors(seq_train, static_train)
    y_tr = torch.from_numpy(y_train.astype(np.float32))
    n_pos = float(y_train.sum())
    pos_weight = torch.tensor((len(y_train) - n_pos) / max(n_pos, 1.0))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_auc, best_state, stale = -np.inf, None, 0
    for epoch in range(cfg.max_epochs):
        model.net.train()
        order = torch.randperm(len(y_tr), generator=gen)
        total = 0.0
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model.net(s_tr[idx], x_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        val_auc = _auc(y_val, model.predict_proba(seq_val, static_val))
        model.history.append(
            {"epoch": epoch + 1, "train_loss": total / len(y_tr), "val_auc": val_auc}
        )
        if val_auc > best_auc:
            best_auc, stale = val_auc, 0
            best_state = {k: v.clone() for k, v in model.net.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.net.load_state_dict(best_state)
    return model


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney rank formula (same value as scikit-learn,
    but fast enough for thousands of bootstrap rounds). 0.5 when only one class
    is present, because there is nothing to rank."""
    y = np.asarray(y)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(score)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    """F1 of the positive class; 0 when there is nothing to score."""
    y, pred = np.asarray(y), np.asarray(pred)
    denom = y.sum() + pred.sum()
    return float(2 * np.sum((y == 1) & (pred == 1)) / denom) if denom else 0.0


@dataclass
class DeepResult:
    """Trained LSTM, its metrics and its test-set outputs."""

    model: LSTMRiskModel
    metrics: pd.DataFrame
    test_pred: np.ndarray
    test_score: np.ndarray


def train_lstm(
    data: OuladData,
    table: pd.DataFrame,
    split: Split,
    cutoff_day: int,
    config: LSTMConfig | None = None,
    seed: int = 42,
) -> DeepResult:
    """Train the LSTM on ``split`` (the same split the baselines used) and score it."""
    if not table.index.equals(pd.RangeIndex(len(table))):
        raise ValueError("table must have a default 0..n-1 index")
    seq = build_sequences(data, table, cutoff_day)
    static = table[STATIC_FEATURES].to_numpy(dtype=float)

    def part(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        i = df.index.to_numpy()
        return seq[i], static[i], df["at_risk"].to_numpy()

    tr, va = part(split.train), part(split.val)
    model = fit_lstm(*tr, *va, config=config, seed=seed)

    rows, test_out = [], None
    for name, df in (("val", split.val), ("test", split.test)):
        s, x, y = part(df)
        score = model.predict_proba(s, x)
        pred = (score >= 0.5).astype(int)
        rows.append({"model": "lstm", "split": name, **evaluate(y, pred, score)})
        if name == "test":
            test_out = (pred, score)
    return DeepResult(model, pd.DataFrame(rows), *test_out)


# -------------------------------------------------------------- significance test


def paired_bootstrap(
    y: np.ndarray,
    groups: np.ndarray,
    out_a: np.ndarray,
    out_b: np.ndarray,
    metric: str = "roc_auc",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Difference ``metric(A) - metric(B)`` on the same test set, with a 95 % CI.

    Students (``groups``) are resampled with replacement, keeping all their
    enrolments together, because rows of one student are not independent.

    ``out_a`` / ``out_b`` are probabilities for ``"roc_auc"`` and 0/1
    predictions for ``"f1"``. The two-sided p-value is the bootstrap estimate
    of how often the sign of the difference flips.
    """
    fns = {"roc_auc": _auc, "f1": _f1}
    if metric not in fns:
        raise ValueError("metric must be 'roc_auc' or 'f1'")
    fn = fns[metric]

    y, out_a, out_b = map(np.asarray, (y, out_a, out_b))
    codes, uniq = pd.factorize(pd.Series(groups))
    members = [np.flatnonzero(codes == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        diffs[b] = fn(y[idx], out_a[idx]) - fn(y[idx], out_b[idx])

    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {
        "a": fn(y, out_a),
        "b": fn(y, out_b),
        "diff": fn(y, out_a) - fn(y, out_b),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value": float(min(p, 1.0)),
    }


def compare_to_baselines(
    test: pd.DataFrame,
    deep: DeepResult,
    baseline_outputs: dict[str, tuple[np.ndarray, np.ndarray]],
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap comparison of the LSTM with every baseline on the test set.

    ``baseline_outputs`` maps a model name to ``(predictions, probabilities)``
    on ``test``. Returns one row per baseline and metric.
    """
    y, groups = test["at_risk"].to_numpy(), test["id_student"].to_numpy()
    rows = []
    for name, (pred, score) in baseline_outputs.items():
        for metric, a, b in (
            ("roc_auc", deep.test_score, score),
            ("f1", deep.test_pred, pred),
        ):
            r = paired_bootstrap(y, groups, a, b, metric=metric, n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "baseline": name,
                    "metric": metric,
                    "lstm": r["a"],
                    "baseline_value": r["b"],
                    "diff": r["diff"],
                    "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"],
                    "p_value": r["p_value"],
                    "significant": r["p_value"] < 0.05,
                }
            )
    return pd.DataFrame(rows)
