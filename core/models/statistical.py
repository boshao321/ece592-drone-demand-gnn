"""Deterministic statistical baselines, fitted on the training split only."""
from __future__ import annotations

import numpy as np

from .. import config as C


class HistoricalAverage:
    """Average demand per (node, time-of-day slot), learned on training data."""

    name = "HA"

    def fit(self, x_train: np.ndarray, tod_train: np.ndarray):
        n = x_train.shape[1]
        self.table = np.zeros((C.SLOTS_PER_DAY, n), dtype=np.float32)
        for s in range(C.SLOTS_PER_DAY):
            m = tod_train == s
            if m.any():
                self.table[s] = x_train[m].mean(0)
        self.fallback = x_train.mean(0)
        return self

    def predict(self, starts, tod):
        out = []
        for s in starts:
            idx = tod[s + C.SEQ_LEN:s + C.SEQ_LEN + C.PRED_LEN]
            out.append(self.table[idx])
        return np.stack(out)


class SeasonalNaive:
    """Demand at the same time of day on the previous day."""

    name = "SeasonalNaive"

    def fit(self, x_full: np.ndarray):
        self.x = x_full
        return self

    def predict(self, starts, tod=None):
        out = []
        for s in starts:
            rows = []
            for h in range(C.PRED_LEN):
                t = s + C.SEQ_LEN + h
                src = t - C.SLOTS_PER_DAY
                rows.append(self.x[src] if src >= 0 else self.x[t % C.SLOTS_PER_DAY])
            out.append(np.stack(rows))
        return np.stack(out)


class RidgeLinear:
    """Ridge regression from the 12 lags to the 4 horizons, weights shared
    across nodes, closed-form solution so it is fully deterministic."""

    name = "Ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    @staticmethod
    def _design(x, starts):
        xs = np.stack([x[s:s + C.SEQ_LEN] for s in starts])        # [S, T, N]
        ys = np.stack([x[s + C.SEQ_LEN:s + C.SEQ_LEN + C.PRED_LEN] for s in starts])
        a = xs.transpose(0, 2, 1).reshape(-1, C.SEQ_LEN)           # node axis first
        b = ys.transpose(0, 2, 1).reshape(-1, C.PRED_LEN)
        return a, b

    def fit(self, x, starts):
        a, b = self._design(x, starts)
        a = np.concatenate([a, np.ones((len(a), 1), np.float32)], 1)
        g = a.T @ a + self.alpha * np.eye(a.shape[1], dtype=np.float64)
        self.w = np.linalg.solve(g, a.T @ b)
        return self

    def predict_from(self, x, starts, n_nodes):
        xs = np.stack([x[s:s + C.SEQ_LEN] for s in starts])
        a = xs.transpose(0, 2, 1).reshape(-1, C.SEQ_LEN)
        a = np.concatenate([a, np.ones((len(a), 1), np.float32)], 1)
        p = a @ self.w
        return p.reshape(len(starts), n_nodes, C.PRED_LEN).transpose(0, 2, 1)
