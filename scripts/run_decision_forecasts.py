"""Experiment (iii), stage 1: forecasts that feed the fleet allocation.

Target: high-value orders (top quartile of order value on days 1-6) pooled over all
business lines, on the same 120-cell grid the pads are defined on. This is the
quantity the allocation maximizes, so forecast target and decision objective
agree.

Every neural model is trained with the L1 loss (q = 0.5). AMSGCN, and AGCRN as
the strongest-baseline control, are also trained with the pinball loss at
q in {0.6, 0.7, 0.8, 0.9}. Validation and test forecasts are saved for every
run; scripts/run_decision_eval.py turns them into fleet plans.

Written for ECE 592. Resumable (existing forecast files are skipped).

    python scripts/run_decision_forecasts.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import config as C                                  # noqa: E402
from core import data as D                                    # noqa: E402
from core.engine import train_model                          # noqa: E402
from core.metrics import per_horizon                          # noqa: E402
from core.runner import build_neural, deterministic_preds, targets   # noqa: E402

FC = C.OUT_ROOT / "forecasts_hv"
TABLES = C.OUT_ROOT / "tables"


def qtag(q):
    return f"q{int(round(q * 100)):02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=C.ALL_MODELS)
    ap.add_argument("--quantile-models", nargs="*", default=["AMSGCN", "AGCRN", "GraphWaveNet", "STAEformer"])
    ap.add_argument("--quantiles", nargs="*", type=float, default=[q for q in C.QUANTILES if q != 0.5])
    ap.add_argument("--seeds", nargs="*", type=int, default=C.SEEDS)
    ap.add_argument("--max-epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--ckpt-dir", default=None, help="save/resume per-epoch training state here")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    FC.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    df = D.load_orders()
    sp = D.build_spatial(df, C.GRID_STEP, include_dropoff=C.INCLUDE_DROPOFF)
    thr = D.high_value_threshold_of(df)
    b = D.build_bundle(df, sp, biz=None, need_dynamic=True, high_value_only=True,
                       high_value_threshold=thr)
    y_val, y_test = targets(b.raw, b.val.starts), targets(b.raw, b.test.starts)
    shared = {"y_val": y_val, "y_test": y_test, "val_starts": b.val.starts,
              "test_starts": b.test.starts, "demand_full": b.raw, "node_coords": sp.coords,
              "high_value_threshold": np.array(thr)}
    for name, arr in shared.items():            # identical for every job; written once
        if not (FC / f"{name}.npy").exists():
            np.save(FC / f"{name}.npy", arr)
    print(f"high-value threshold {thr:.2f} (days 1-6); target on {sp.n} cells; test mean {y_test.mean():.3f}, "
          f"nonzero {(y_test > 0).mean() * 100:.1f}%")

    acc_path = TABLES / f"decision_forecast_accuracy_{args.tag}.csv"
    rows = pd.read_csv(acc_path).to_dict("records") if acc_path.exists() else []

    jobs = []
    for m in args.models:
        if m in C.DETERMINISTIC_MODELS:
            jobs.append((m, "det", None, -1))
        else:
            jobs += [(m, "l1", 0.5, s) for s in args.seeds]
    for m in args.quantile_models:
        for q in args.quantiles:
            jobs += [(m, "pinball", q, s) for s in args.seeds]

    for name, loss, q, seed in jobs:
        f = FC / (f"pred_{name}_det.npz" if loss == "det" else f"pred_{name}_{qtag(q)}_s{seed}.npz")
        if f.exists():
            continue
        if loss == "det":
            pv = deterministic_preds(name, b.raw, b.val.starts)
            pt = deterministic_preds(name, b.raw, b.test.starts)
            m = per_horizon(pt.astype(np.float64), y_test.astype(np.float64))
        else:
            C.seed_everything(seed)
            model = build_neural(name, b)
            ck = None
            if args.ckpt_dir:
                Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                ck = Path(args.ckpt_dir) / (f.stem + ".pt")
            m, pt, _, pv, _ = train_model(model, b, seed, max_epochs=args.max_epochs,
                                          loss="l1" if loss == "l1" else "pinball", q=q, ckpt_path=ck)
            if ck is not None:
                ck.unlink(missing_ok=True)
        np.savez_compressed(f, p_val=pv.astype(np.float32), p_test=pt.astype(np.float32))
        rows.append({"model": name, "loss": loss, "q": q, "seed": seed, **m})
        pd.DataFrame(rows).to_csv(acc_path, index=False)
        print(f"  {name:<13} {loss:<7} q={q}  seed {seed}  MAE {m['MAE']:.4f}  "
              f"MAE@t+60 {m['MAE@t+60']:.4f}  ({m.get('epochs', 0)} ep, {m.get('train_sec', 0):.0f}s)",
              flush=True)
    print(f"\nforecasts in {FC}")


if __name__ == "__main__":
    main()
