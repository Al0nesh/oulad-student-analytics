"""Sequence models and a common trainer (selection on validation ROC-AUC only)."""

import copy

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

torch.set_num_threads(2)


class StaticBranch(nn.Module):
    def __init__(self, n_static, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_static, out), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, d_in, dropout, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class RNNNet(nn.Module):
    """kind: lstm | gru | bilstm | lstm_attn"""

    def __init__(self, kind, n_ch, n_static, hidden=64, layers=1, dropout=0.2, static_hidden=32):
        super().__init__()
        self.kind = kind
        rnn = nn.GRU if kind == "gru" else nn.LSTM
        bi = kind == "bilstm"
        self.rnn = rnn(
            n_ch,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bi,
            dropout=dropout if layers > 1 else 0.0,
        )
        d = hidden * (2 if bi else 1)
        if kind == "lstm_attn":
            self.attn = nn.Linear(d, 1)
            d_seq = 2 * d  # attention summary + last state
        else:
            d_seq = d
        self.static = StaticBranch(n_static, static_hidden)
        self.head = Head(d_seq + static_hidden, dropout)

    def forward(self, s, x, return_attn=False):
        out, h = self.rnn(s)
        h = h[0] if isinstance(h, tuple) else h
        last = torch.cat([h[-2], h[-1]], dim=1) if self.kind == "bilstm" else h[-1]
        attn = None
        if self.kind == "lstm_attn":
            attn = torch.softmax(self.attn(out).squeeze(-1), dim=1)
            summary = (attn.unsqueeze(-1) * out).sum(1)
            last = torch.cat([last, summary], dim=1)
        y = self.head(torch.cat([last, self.static(x)], dim=1))
        return (y, attn) if return_attn else y


class CNNNet(nn.Module):
    def __init__(
        self, n_ch, n_static, hidden=64, layers=2, dropout=0.2, static_hidden=32, kernel=3
    ):
        super().__init__()
        blocks, c = [], n_ch
        for _ in range(layers):
            blocks += [
                nn.Conv1d(c, hidden, kernel, padding=kernel // 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ]
            c = hidden
        self.conv = nn.Sequential(*blocks)
        self.static = StaticBranch(n_static, static_hidden)
        self.head = Head(2 * hidden + static_hidden, dropout)

    def forward(self, s, x):
        z = self.conv(s.transpose(1, 2))
        pooled = torch.cat([z.mean(2), z.amax(2)], dim=1)
        return self.head(torch.cat([pooled, self.static(x)], dim=1))


class TransformerNet(nn.Module):
    def __init__(
        self,
        n_ch,
        n_static,
        hidden=64,
        layers=2,
        dropout=0.2,
        static_hidden=32,
        heads=4,
        max_len=64,
    ):
        super().__init__()
        self.inp = nn.Linear(n_ch, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        enc = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=2 * hidden, dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc, layers)
        self.static = StaticBranch(n_static, static_hidden)
        self.head = Head(2 * hidden + static_hidden, dropout)

    def forward(self, s, x):
        z = self.enc(self.inp(s) + self.pos[:, : s.shape[1]])
        pooled = torch.cat([z.mean(1), z[:, -1]], dim=1)
        return self.head(torch.cat([pooled, self.static(x)], dim=1))


def make_net(kind, n_ch, n_static, **kw):
    if kind in ("lstm", "gru", "bilstm", "lstm_attn"):
        return RNNNet(
            kind, n_ch, n_static, **{k: v for k, v in kw.items() if k != "heads" and k != "kernel"}
        )
    if kind == "cnn":
        return CNNNet(n_ch, n_static, **{k: v for k, v in kw.items() if k != "heads"})
    if kind == "transformer":
        kw = {k: v for k, v in kw.items() if k != "kernel"}
        if kw.get("hidden", 64) % kw.get("heads", 4):
            kw["heads"] = 2
        return TransformerNet(n_ch, n_static, **kw)
    raise ValueError(kind)


class Scaler:
    def __init__(self, seq_tr, static_tr):
        flat = seq_tr.reshape(-1, seq_tr.shape[-1])
        self.sm, self.ss = flat.mean(0), flat.std(0) + 1e-6
        med = np.nanmedian(static_tr, axis=0)
        self.med = np.where(np.isnan(med), 0, med)
        f = np.where(np.isnan(static_tr), self.med, static_tr)
        self.xm, self.xs = f.mean(0), f.std(0) + 1e-6

    def __call__(self, seq, static):
        s = (seq - self.sm) / self.ss
        x = (np.where(np.isnan(static), self.med, static) - self.xm) / self.xs
        return torch.from_numpy(s.astype(np.float32)), torch.from_numpy(x.astype(np.float32))


def predict(net, s, x, bs=4096):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(s), bs):
            out.append(torch.sigmoid(net(s[i : i + bs], x[i : i + bs])))
    return torch.cat(out).numpy()


def train(
    kind,
    seq,
    static,
    y,
    tr,
    va,
    *,
    hidden=64,
    layers=1,
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    batch=256,
    static_hidden=32,
    heads=4,
    kernel=3,
    max_epochs=80,
    patience=10,
    seed=42,
    pos_weight=True,
    verbose=False,
):
    """Train one network with early stopping on validation ROC-AUC.

    Returns (net, scaler, best_auc, epochs).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    sc = Scaler(seq[tr], static[tr])
    s_tr, x_tr = sc(seq[tr], static[tr])
    s_va, x_va = sc(seq[va], static[va])
    y_tr = torch.from_numpy(y[tr].astype(np.float32))
    net = make_net(
        kind,
        seq.shape[2],
        static.shape[1],
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        static_hidden=static_hidden,
        heads=heads,
        kernel=kernel,
    )
    pw = torch.tensor((len(y_tr) - y_tr.sum().item()) / y_tr.sum().item()) if pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    best, best_state, stale, ep = -1.0, None, 0, 0
    for ep in range(1, max_epochs + 1):
        net.train()
        order = torch.randperm(len(y_tr), generator=gen)
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(net(s_tr[idx], x_tr[idx]), y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        auc = roc_auc_score(y[va], predict(net, s_va, x_va))
        sched.step(auc)
        if verbose:
            print(ep, round(auc, 4), flush=True)
        if auc > best + 1e-5:
            best, best_state, stale = auc, copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    net.load_state_dict(best_state)
    return net, sc, best, ep


def predict_on(net, sc, seq, static, idx):
    s, x = sc(seq[idx], static[idx])
    return predict(net, s, x)
