"""Experiment (i): forecasting benchmark on three business lines.

Every model forecasts 15-minute order counts on the 120-cell union grid for
t+15 ... t+60. Statistical baselines run once; neural models run once per seed
and are summarized as mean +/- s.d. by scripts/summarize.py.

Written for ECE 592. Resumable: finished (business line, model, seed) runs are
read back from disk and skipped.

    python scripts/run_benchmark.py                      # everything
    python scripts/run_benchmark.py --biz waimai --tag a # one line, own output file
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

TABLES = C.OUT_ROOT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--biz", nargs="*", default=C.BIZ_LINES)
    ap.add_argument("--models", nargs="*", default=C.ALL_MODELS)
    ap.add_argument("--seeds", nargs="*", type=int, default=C.SEEDS)
    ap.add_argument("--max-epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--ckpt-dir", default=None, help="save/resume per-epoch training state here")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    out_path = TABLES / f"benchmark_runs_{args.tag}.csv"
    rows, done = [], set()
    if out_path.exists():
        rows = pd.read_csv(out_path).to_dict("records")
        done = {(r["biz"], r["model"], int(r["seed"])) for r in rows}
        print(f"resuming: {len(rows)} runs on disk")

    def flush():
        pd.DataFrame(rows).to_csv(out_path, index=False)

    df = D.load_orders()
    sp = D.build_spatial(df, C.GRID_STEP, include_dropoff=C.INCLUDE_DROPOFF)
    print(f"{len(df):,} orders; {sp.n} cells at {C.GRID_STEP} deg")

    for biz in args.biz:
        b = D.build_bundle(df, sp, biz, need_dynamic=True)
        zb = b.zero_baseline
        print(f"\n== {biz}: windows {len(b.train)}/{len(b.val)}/{len(b.test)}, "
              f"test mean {zb['test_mean_demand']:.3f}, nonzero {zb['test_nonzero_share_pct']:.1f}%")
        if (biz, "ZeroBaseline", -1) not in done:
            y = targets(b.raw, b.test.starts)
            m = per_horizon(np.zeros_like(y, dtype=np.float64), y.astype(np.float64))
            rows.append({"biz": biz, "model": "ZeroBaseline", "seed": -1, **m})
            flush()
        for name in args.models:
            if name in C.DETERMINISTIC_MODELS:
                if (biz, name, -1) in done:
                    continue
                p = deterministic_preds(name, b.raw, b.test.starts)
                y = targets(b.raw, b.test.starts)
                m = per_horizon(p.astype(np.float64), y.astype(np.float64))
                rows.append({"biz": biz, "model": name, "seed": -1, **m})
                flush()
                print(f"  {name:<13} MAE {m['MAE']:.4f}  MAE@t+60 {m['MAE@t+60']:.4f}")
                continue
            for seed in args.seeds:
                if (biz, name, seed) in done:
                    continue
                C.seed_everything(seed)
                model = build_neural(name, b)
                ck = None
                if args.ckpt_dir:
                    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                    ck = Path(args.ckpt_dir) / f"bench_{biz}_{name}_{seed}.pt"
                m, *_ = train_model(model, b, seed, max_epochs=args.max_epochs, ckpt_path=ck)
                rows.append({"biz": biz, "model": name, "seed": seed, **m})
                flush()
                if ck is not None:
                    ck.unlink(missing_ok=True)
                print(f"  {name:<13} seed {seed}  MAE {m['MAE']:.4f}  MAE@t+60 {m['MAE@t+60']:.4f}  "
                      f"({m['epochs']} ep, {m['train_sec']:.0f}s)", flush=True)
    flush()
    print(f"\nwritten {out_path}")


if __name__ == "__main__":
    main()
