"""Per-horizon error metrics on the original demand scale.

MAPE is deliberately absent: demand is sparse and the denominator is zero in a
large share of cells, which makes MAPE both undefined and unstable. WAPE keeps
the same relative reading without that problem.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def per_horizon(preds: np.ndarray, labels: np.ndarray) -> dict:
    """preds, labels: [S, PRED_LEN, N]. Returns flat metrics per horizon plus
    an average over horizons."""
    out = {}
    for h in range(preds.shape[1]):
        p, y = preds[:, h], labels[:, h]
        err = p - y
        mae = float(np.abs(err).mean())
        rmse = float(np.sqrt((err ** 2).mean()))
        denom = float(np.abs(y).sum())
        wape = float(np.abs(err).sum() / denom * 100) if denom > 0 else float("nan")
        label = C.HORIZON_LABELS[h]
        out[f"MAE@{label}"] = mae
        out[f"RMSE@{label}"] = rmse
        out[f"WAPE@{label}"] = wape
    err = preds - labels
    out["MAE"] = float(np.abs(err).mean())
    out["RMSE"] = float(np.sqrt((err ** 2).mean()))
    denom = float(np.abs(labels).sum())
    out["WAPE"] = float(np.abs(err).sum() / denom * 100) if denom > 0 else float("nan")
    return out
