"""Experiment (iii), stage 2: fleet allocation and gap closed (GC).

For each saved forecast, the t+60 prediction is aggregated to the 38 pads,
converted to aircraft-minutes, and passed to the exact greedy allocator for
fleets M in {50, 100, 200, 300}. The plan is then scored against the realized
high-value orders. GC is the share of the static-to-oracle gap a forecast closes:

    GC = (served(forecast) - served(static)) / (served(oracle) - served(static))

GC is computed on the validation day (day 6) and on the test day (day 7). For
the pinball-trained models the quantile q is selected per fleet size on the
validation day, and only its test-day GC is reported as the method's result.

Written for ECE 592.

    python scripts/run_decision_eval.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import allocation as A          # noqa: E402
from core import config as C              # noqa: E402
from core import data as D                # noqa: E402

FC = C.OUT_ROOT / "forecasts_hv"
TABLES = C.OUT_ROOT / "tables"
SCALE_GRID = [round(1.0 + 0.1 * i, 1) for i in range(16)]     # k = 1.0 ... 2.5
PAT = re.compile(r"pred_(?P<model>[A-Za-z]+)_(?:(?P<det>det)|q(?P<q>\d+)_s(?P<seed>\d+))\.npz")


def load_sites(coords):
    sel = pd.read_csv(C.SITES_CSV)
    index = {(round(a, 6), round(b, 6)): i for i, (a, b) in enumerate(coords)}
    idx = [index[(round(float(a), 6), round(float(b), 6))] for a, b in zip(sel["lat"], sel["lng"])
           if (round(float(a), 6), round(float(b), 6)) in index]
    return np.array(sorted(set(idx)))


def prepare_orders():
    df = D.load_orders().copy()
    df["p_lat_snap"] = D.snap(df["pickup_latitude"], C.GRID_STEP)
    df["p_lng_snap"] = D.snap(df["pickup_longitude"], C.GRID_STEP)
    df["od_km"] = A.haversine_km(df["pickup_latitude"].to_numpy(), df["pickup_longitude"].to_numpy(),
                                 df["dropoff_latitude"].to_numpy(), df["dropoff_longitude"].to_numpy())
    return df


def main():
    coords = np.load(FC / "node_coords.npy")
    demand_full = np.load(FC / "demand_full.npy")
    h = C.DECISION_HORIZON
    site_idx = load_sites(coords)
    orders = prepare_orders()
    sp = D.build_spatial(D.load_orders(), C.GRID_STEP, include_dropoff=C.INCLUDE_DROPOFF)
    assert np.allclose(sp.coords, coords)
    grid_to_site, _ = A.assign_grids_to_sites(coords, site_idx)
    train_slots = np.arange(0, C.TRAIN_DAYS * C.SLOTS_PER_DAY)

    A.HIGH_VALUE_THRESHOLD = float(np.load(FC / "high_value_threshold.npy"))
    print(f"high-value threshold {A.HIGH_VALUE_THRESHOLD:.2f} (days 1-6)")
    tr = orders[orders["slot"].isin(train_slots) & (orders["od_km"] <= A.MAX_SORTIE_RANGE_KM)]
    tr_hv = tr[tr["package_price"] >= A.HIGH_VALUE_THRESHOLD]
    min_per_order = float(A.sortie_minutes(tr_hv["od_km"].to_numpy()).mean())
    print(f"{len(site_idx)} pads; {int((grid_to_site >= 0).sum())} of {len(coords)} cells within "
          f"{A.SERVICE_RADIUS_KM} km; {min_per_order:.2f} aircraft-min per high-value sortie")

    splits, scaled_rows = {}, []
    for split in ("val", "test"):
        starts = np.load(FC / f"{split}_starts.npy")
        slots = starts + C.SEQ_LEN + h                      # the t+60 target slot of each window
        sc = A.build_scenario(orders, coords, site_idx, slots, grid_index=sp.index)
        est = A.policy_estimates(sc, demand_full, grid_to_site, np.zeros_like(sc.req_minutes),
                                 train_slots, min_per_order, lead_slots=h + 1)
        ref = {}
        for M in C.FLEET_SIZES:
            ref[M] = {pol: A.evaluate(A.allocate(est[pol], M), sc)["served_high_value_share_pct"]
                      for pol in ("static", "lagged", "oracle")}
        # no-learning control: the static plan inflated by a safety factor k
        for k in SCALE_GRID:
            for M in C.FLEET_SIZES:
                served = A.evaluate(A.allocate(est["static"] * k, M), sc)["served_high_value_share_pct"]
                s_, o_ = ref[M]["static"], ref[M]["oracle"]
                scaled_rows.append({"split": split, "fleet": M, "model": "StaticScaled", "loss": "scaled",
                                    "q": k, "seed": -1, "served_hv_pct": served, "static_pct": s_,
                                    "lagged_pct": ref[M]["lagged"], "oracle_pct": o_,
                                    "GC_pct": 100.0 * (served - s_) / (o_ - s_)})
        splits[split] = (sc, ref)
        print(f"  {split}: {sc.req_orders_hv.sum():,.0f} air-servable high-value orders; "
              + "; ".join(f"M={M}: static {r['static']:.1f}% oracle {r['oracle']:.1f}%"
                          for M, r in ref.items()))

    rows = []
    files = sorted(FC.glob("pred_*.npz"))
    for f in files:
        mt = PAT.fullmatch(f.name)
        if not mt:
            continue
        model = mt["model"]
        if mt["det"]:
            q, seed, loss = np.nan, -1, "det"
        else:
            q, seed = int(mt["q"]) / 100, int(mt["seed"])
            loss = "l1" if q == 0.5 else "pinball"
        z = np.load(f)
        for split, key in (("val", "p_val"), ("test", "p_test")):
            sc, ref = splits[split]
            pred_sites = A.nodes_to_sites(z[key][:, h, :], grid_to_site, sc.n_sites)
            for M in C.FLEET_SIZES:
                served = A.evaluate(A.allocate(pred_sites * min_per_order, M), sc)["served_high_value_share_pct"]
                s, o = ref[M]["static"], ref[M]["oracle"]
                rows.append({"split": split, "fleet": M, "model": model, "loss": loss, "q": q,
                             "seed": seed, "served_hv_pct": served, "static_pct": s,
                             "lagged_pct": ref[M]["lagged"], "oracle_pct": o,
                             "GC_pct": 100.0 * (served - s) / (o - s)})
    runs = pd.DataFrame(rows + scaled_rows)
    runs.to_csv(TABLES / "decision_runs.csv", index=False)

    # reference policies as rows too
    refs = [{"split": sp_, "fleet": M, "policy": pol, "served_hv_pct": v}
            for sp_, (_, ref) in splits.items() for M, r in ref.items() for pol, v in r.items()]
    pd.DataFrame(refs).to_csv(TABLES / "decision_reference_policies.csv", index=False)

    # ---- summary: every (model, q) on test, mean +/- s.d. over seeds
    g = runs.groupby(["split", "fleet", "model", "loss", "q"], dropna=False)["GC_pct"]
    summ = g.agg(["mean", "std", "count"]).reset_index()
    summ.to_csv(TABLES / "decision_gc_by_q.csv", index=False)

    # ---- validation-selected quantile per (model, fleet), reported on test
    sel_rows = []
    for (model, M), gg in summ[summ.split == "val"].groupby(["model", "fleet"]):
        cand = gg[gg.loss.isin(["l1", "pinball", "scaled"])]
        if len(cand) < 2:
            continue
        best = cand.loc[cand["mean"].idxmax()]
        t = summ[(summ.split == "test") & (summ.model == model) & (summ.fleet == M)
                 & (summ.q.round(2) == round(best.q, 2))].iloc[0]
        t50 = summ[(summ.split == "test") & (summ.model == model) & (summ.fleet == M)
                   & (summ.loss == "l1")]
        sel_rows.append({"model": model, "fleet": M, "q_selected": best.q,
                         "val_GC_selected": best["mean"],
                         "test_GC_selected_mean": t["mean"], "test_GC_selected_std": t["std"],
                         "test_GC_l1_mean": t50["mean"].iloc[0] if len(t50) else np.nan,
                         "test_GC_l1_std": t50["std"].iloc[0] if len(t50) else np.nan,
                         "n_seeds": int(t["count"])})
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(TABLES / "decision_quantile_selected.csv", index=False)

    print("\nTest-day GC (%), L1-trained models, mean over seeds:")
    piv = summ[(summ.split == "test") & (summ.loss.isin(["l1", "det"]))].pivot_table(
        index="model", columns="fleet", values="mean")
    print(piv.round(1).to_string())
    if len(sel):
        print("\nValidation-selected quantile, test-day GC (%):")
        print(sel.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
