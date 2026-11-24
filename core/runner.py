"""Model construction and deterministic baselines (written for ECE 592).

Keeps the experiment scripts short: every script builds models through
`build_neural` and gets statistical forecasts through `deterministic_preds`.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .models import baselines as B
from .models.amsgcn import AMSGCN
from .models.transformer import STAEformer
from .models.statistical import HistoricalAverage, SeasonalNaive

TOD = (np.arange(C.TOTAL_SLOTS) % C.SLOTS_PER_DAY).astype(np.int64)


def build_neural(name, bundle, **amsgcn_kwargs):
    n = bundle.n_nodes
    if name == "LSTM":
        return B.LSTMBaseline(n)
    if name == "STGCN":
        return B.STGCN(n, bundle.adj_static)
    if name == "DCRNN":
        return B.DCRNN(n, bundle.adj_static)
    if name == "GraphWaveNet":
        return B.GraphWaveNet(n, bundle.adj_static)
    if name == "AGCRN":
        return B.AGCRN(n)
    if name == "AMSGCN":
        return AMSGCN(n, **amsgcn_kwargs)
    if name == "STAEformer":
        return STAEformer(n)
    raise ValueError(name)


def deterministic_preds(name, x_raw, starts):
    """Forecasts for the given window starts, on the original scale."""
    if name == "HA":
        train_slots = np.arange(0, C.TRAIN_DAYS * C.SLOTS_PER_DAY)
        m = HistoricalAverage().fit(x_raw[train_slots], TOD[train_slots])
        return m.predict(starts, TOD)
    if name == "SeasonalNaive":
        return SeasonalNaive().fit(x_raw).predict(starts)
    raise ValueError(name)


def targets(x_raw, starts):
    return np.stack([x_raw[s + C.SEQ_LEN:s + C.SEQ_LEN + C.PRED_LEN] for s in starts])
