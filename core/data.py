"""Data pipeline: unified node set, demand tensor, OD flow graphs, splits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config as C


# ------------------------------------------------------------------ helpers


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def _to_minutes(series: pd.Series) -> pd.Series:
    parts = series.astype(str).str.split(":", expand=True)
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else 0
    return h * 60 + m


def load_orders() -> pd.DataFrame:
    df = pd.read_csv(C.RAW_CSV, encoding="utf-8")
    df["pay_min"] = _to_minutes(df["customer_pay_time"])
    df = df.dropna(subset=["pay_min", "pickup_latitude", "pickup_longitude",
                           "dropoff_latitude", "dropoff_longitude", "biz_line"])
    df["slot"] = ((df["dt_seq_id"] - 1) * C.SLOTS_PER_DAY
                  + (df["pay_min"] // C.RESOLUTION_MIN)).astype(int)
    df = df[(df["slot"] >= 0) & (df["slot"] < C.TOTAL_SLOTS)]
    return df


def high_value_threshold_of(df: pd.DataFrame) -> float:
    """75th percentile of order value over days 1-6 only.

    The test day (day 7) is excluded so that the definition of a high-value
    order uses no information from the period being scored.
    """
    pre = df[df["slot"] < (C.TRAIN_DAYS + C.VAL_DAYS) * C.SLOTS_PER_DAY]
    return float(np.quantile(pre["package_price"].to_numpy(dtype=float), C.HIGH_VALUE_QUANTILE))


def snap(values, step):
    return np.round(np.round(np.asarray(values, dtype=float) / step) * step, 6)


@dataclass
class Spatial:
    """One node set, shared by every business line at a given grid step."""
    step: float
    coords: np.ndarray              # [N, 2] lat, lng
    index: dict                     # (lat, lng) -> node id

    @property
    def n(self) -> int:
        return len(self.coords)

    def ids(self, lat, lng) -> np.ndarray:
        keys = list(zip(snap(lat, self.step), snap(lng, self.step)))
        return np.array([self.index.get(k, -1) for k in keys], dtype=np.int64)

    def distance_adjacency(self) -> np.ndarray:
        la, lo = self.coords[:, 0], self.coords[:, 1]
        d = haversine_km(la[:, None], lo[:, None], la[None, :], lo[None, :])
        w = np.exp(-(d ** 2) / (C.ADJ_SIGMA_KM ** 2))
        w[w < C.ADJ_THRESHOLD] = 0.0
        np.fill_diagonal(w, 1.0)
        return w.astype(np.float32)


def build_spatial(df: pd.DataFrame, step: float, include_dropoff: bool = False) -> Spatial:
    """Node set pooled over ALL business lines, so a node index means the same
    place for every line and every model.

    `include_dropoff=True` unions pickup and dropoff grids, which is the node
    set the siting stage uses. Running the forecasting stage on the same union
    is what lets a forecast be handed directly to the allocation stage: with
    pickup grids only the two stages would be defined over different universes,
    27 of the siting candidates at 0.010 degrees never appearing in the
    forecasting node set at all.
    """
    lat = list(snap(df["pickup_latitude"], step))
    lng = list(snap(df["pickup_longitude"], step))
    if include_dropoff:
        lat += list(snap(df["dropoff_latitude"], step))
        lng += list(snap(df["dropoff_longitude"], step))
    keys = sorted(set(zip(lat, lng)))
    index = {k: i for i, k in enumerate(keys)}
    coords = np.array(keys, dtype=float)
    return Spatial(step=step, coords=coords, index=index)


def demand_tensor(df: pd.DataFrame, sp: Spatial, biz: str | None,
                  high_value_only: bool = False,
                  high_value_threshold: float | None = None) -> np.ndarray:
    """Order originations per grid per slot.

    `high_value_only` restricts the count to orders at or above the 75th
    percentile of order value, which is the same definition the siting stage
    uses. Forecasting that quantity rather than total demand is what makes the
    prediction target match the objective the allocation stage optimizes.
    """
    sub = df if biz is None else df[df["biz_line"] == biz]
    if high_value_only:
        thr = high_value_threshold if high_value_threshold is not None else high_value_threshold_of(df)
        sub = sub[sub["package_price"] >= thr]
    x = np.zeros((C.TOTAL_SLOTS, sp.n), dtype=np.float32)
    nid = sp.ids(sub["pickup_latitude"], sub["pickup_longitude"])
    ok = nid >= 0
    np.add.at(x, (sub["slot"].to_numpy()[ok], nid[ok]), 1.0)
    return x


def od_flow_per_slot(df: pd.DataFrame, sp: Spatial, biz: str | None):
    """Observed origin-destination counts per time slot, kept sparse."""
    sub = df if biz is None else df[df["biz_line"] == biz]
    o = sp.ids(sub["pickup_latitude"], sub["pickup_longitude"])
    d = sp.ids(sub["dropoff_latitude"], sub["dropoff_longitude"])
    s = sub["slot"].to_numpy()
    ok = (o >= 0) & (d >= 0)
    o, d, s = o[ok], d[ok], s[ok]
    key = (s.astype(np.int64) * sp.n + o) * sp.n + d
    uniq, cnt = np.unique(key, return_counts=True)
    su = uniq // (sp.n * sp.n)
    ou = (uniq // sp.n) % sp.n
    du = uniq % sp.n
    return su.astype(np.int32), ou.astype(np.int32), du.astype(np.int32), cnt.astype(np.float32)


def window_flow_adjacency(flow, sp: Spatial, n_windows: int) -> np.ndarray:
    """Row-normalized OD adjacency for each input window [i, i+SEQ_LEN).

    The matrix changes with t because it is built from the flows actually
    observed inside that window, which is what makes the graph dynamic.
    """
    s, o, d, c = flow
    per_slot = np.zeros((C.TOTAL_SLOTS, sp.n, sp.n), dtype=np.float32)
    per_slot[s, o, d] = c
    cum = np.concatenate([np.zeros((1, sp.n, sp.n), np.float32),
                          np.cumsum(per_slot, axis=0)], axis=0)
    out = np.empty((n_windows, sp.n, sp.n), dtype=np.float32)
    for i in range(n_windows):
        a = cum[i + C.SEQ_LEN] - cum[i]
        a = a + a.T                      # treat the corridor as undirected capacity
        np.fill_diagonal(a, a.diagonal() + 1.0)
        a /= a.sum(axis=1, keepdims=True) + 1e-8
        out[i] = a
    return out


class StandardScaler:
    def __init__(self, mean: float, std: float):
        self.mean, self.std = float(mean), float(std)

    def transform(self, x):
        return (x - self.mean) / (self.std + 1e-8)

    def inverse(self, x):
        return x * (self.std + 1e-8) + self.mean


class WindowDataset(Dataset):
    """Samples are windows over the full series; the split decides which
    window START indices belong to train, validation and test, so no window
    ever straddles a split boundary."""

    def __init__(self, x_norm, starts, tod, dow, adj_dyn):
        self.x = x_norm
        self.starts = starts
        self.tod = tod
        self.dow = dow
        self.adj = adj_dyn

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        xi = self.x[s:s + C.SEQ_LEN]                       # [T, N]
        yi = self.x[s + C.SEQ_LEN:s + C.SEQ_LEN + C.PRED_LEN]
        item = {
            "x": torch.from_numpy(xi).unsqueeze(-1),        # [T, N, 1]
            "y": torch.from_numpy(yi).unsqueeze(-1),        # [P, N, 1]
            "tod": torch.from_numpy(self.tod[s:s + C.SEQ_LEN + C.PRED_LEN]),
            "dow": torch.from_numpy(self.dow[s:s + C.SEQ_LEN + C.PRED_LEN]),
        }
        item["adj"] = torch.from_numpy(self.adj[s]) if self.adj is not None else torch.zeros(1)
        return item


@dataclass
class Bundle:
    train: WindowDataset
    val: WindowDataset
    test: WindowDataset
    scaler: StandardScaler
    adj_static: np.ndarray
    n_nodes: int
    zero_baseline: dict          # MAE/RMSE of always predicting zero, on test
    raw: np.ndarray


def build_bundle(df: pd.DataFrame, sp: Spatial, biz: str | None, need_dynamic: bool,
                 high_value_only: bool = False, high_value_threshold: float | None = None) -> Bundle:
    x = demand_tensor(df, sp, biz, high_value_only=high_value_only,
                      high_value_threshold=high_value_threshold)        # [T, N]
    n_windows = C.TOTAL_SLOTS - C.SEQ_LEN - C.PRED_LEN + 1

    slot_idx = np.arange(C.TOTAL_SLOTS)
    tod = (slot_idx % C.SLOTS_PER_DAY).astype(np.int64)
    dow = (slot_idx // C.SLOTS_PER_DAY).astype(np.int64)

    # Split by whole days rather than by a fraction of the slot index.
    #
    # A 7:2:1 cut of the slot index puts the test set inside a single daytime
    # stretch: with seven days it covers 07:00 to 23:45 of the last day and
    # contains no night hours at all, while the validation set is 36% night.
    # Selecting a model on that validation set and scoring it on that test set
    # compares two different times of day. Cutting on day boundaries gives
    # every split a complete 24-hour cycle.
    #
    # A window's INPUT may reach back into the previous split, which is legitimate
    # because that history is observed before prediction time; only the TARGET
    # slots are required to lie inside the split.
    def starts_for(day_lo, day_hi):
        first_target = day_lo * C.SLOTS_PER_DAY
        last_target_start = day_hi * C.SLOTS_PER_DAY - C.PRED_LEN
        lo = max(0, first_target - C.SEQ_LEN)
        hi = last_target_start - C.SEQ_LEN
        return np.arange(lo, max(lo, hi + 1))

    tr = starts_for(0, C.TRAIN_DAYS)
    va = starts_for(C.TRAIN_DAYS, C.TRAIN_DAYS + C.VAL_DAYS)
    te = starts_for(C.TRAIN_DAYS + C.VAL_DAYS, C.N_DAYS)
    train_end = C.TRAIN_DAYS * C.SLOTS_PER_DAY

    scaler = StandardScaler(x[:train_end].mean(), x[:train_end].std())
    x_norm = scaler.transform(x)

    adj_dyn = window_flow_adjacency(od_flow_per_slot(df, sp, biz), sp, n_windows) if need_dynamic else None

    y_true = np.stack([x[s + C.SEQ_LEN:s + C.SEQ_LEN + C.PRED_LEN] for s in te]) if len(te) else np.zeros((0, C.PRED_LEN, sp.n))
    zero_baseline = {
        "MAE": float(np.abs(y_true).mean()),
        "RMSE": float(np.sqrt((y_true ** 2).mean())),
        "test_mean_demand": float(y_true.mean()),
        "test_nonzero_share_pct": float((y_true > 0).mean() * 100),
    }

    mk = lambda idx: WindowDataset(x_norm, idx, tod, dow, adj_dyn)
    return Bundle(mk(tr), mk(va), mk(te), scaler, sp.distance_adjacency(),
                  sp.n, zero_baseline, x)
