"""Experiment (ii): AMSGCN ablation on three business lines.

Each variant removes one component for real (see tests/test_node_axis.py).
The full model is included so the script is self-contained.

Written for ECE 592. Resumable.

    python scripts/run_ablation.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import config as C                   # noqa: E402
from core import data as D                     # noqa: E402
from core.engine import train_model           # noqa: E402
from core.runner import build_neural           # noqa: E402

TABLES = C.OUT_ROOT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--biz", nargs="*", default=C.BIZ_LINES)
    ap.add_argument("--variants", nargs="*", default=list(C.ABLATION_VARIANTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=C.SEEDS)
    ap.add_argument("--max-epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--ckpt-dir", default=None, help="save/resume per-epoch training state here")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    out_path = TABLES / f"ablation_runs_{args.tag}.csv"
    rows, done = [], set()
    if out_path.exists():
        rows = pd.read_csv(out_path).to_dict("records")
        done = {(r["biz"], r["variant"], int(r["seed"])) for r in rows}

    df = D.load_orders()
    sp = D.build_spatial(df, C.GRID_STEP, include_dropoff=C.INCLUDE_DROPOFF)
    for biz in args.biz:
        b = D.build_bundle(df, sp, biz, need_dynamic=True)
        print(f"\n== {biz}")
        for var in args.variants:
            kw = C.ABLATION_VARIANTS[var]
            for seed in args.seeds:
                if (biz, var, seed) in done:
                    continue
                C.seed_everything(seed)
                model = build_neural("AMSGCN", b, **kw)
                ck = None
                if args.ckpt_dir:
                    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                    ck = Path(args.ckpt_dir) / f"abl_{biz}_{var.replace(' ', '_').replace('/', '')}_{seed}.pt"
                m, *_ = train_model(model, b, seed, max_epochs=args.max_epochs, ckpt_path=ck)
                rows.append({"biz": biz, "variant": var, "seed": seed, **m})
                pd.DataFrame(rows).to_csv(out_path, index=False)
                if ck is not None:
                    ck.unlink(missing_ok=True)
                print(f"  {var:<26} seed {seed}  MAE {m['MAE']:.4f}  MAE@t+60 {m['MAE@t+60']:.4f}"
                      f"  ({m['epochs']} ep, {m['train_sec']:.0f}s)", flush=True)
    print(f"\nwritten {out_path}")


if __name__ == "__main__":
    main()
