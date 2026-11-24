"""Run every experiment of the report as independent single-run jobs in parallel.

Each job is one (experiment, business line, model or variant, seed[, quantile])
and writes its own results file (unique --tag), so jobs never share an output
file. GPUs are assigned round-robin through CUDA_VISIBLE_DEVICES. Finished jobs
are skipped on a rerun because every script is resumable.

Written for ECE 592.

    python scripts/launch_parallel.py --jobs 32 --gpus 0 1
    python scripts/launch_parallel.py --only decision --jobs 16
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import config as C          # noqa: E402

PY = sys.executable


def benchmark_jobs(models, with_det):
    jobs = [["run_benchmark.py", "--models", *C.DETERMINISTIC_MODELS, "--tag", "det"]] if with_det else []
    for biz in C.BIZ_LINES:
        for m in models:
            for s in C.SEEDS:
                jobs.append(["run_benchmark.py", "--biz", biz, "--models", m, "--seeds", str(s),
                             "--tag", f"{biz}_{m}_{s}"])
    return jobs


def ablation_jobs():
    jobs = []
    for biz in C.BIZ_LINES:
        for i, v in enumerate(C.ABLATION_VARIANTS):
            for s in C.SEEDS:
                jobs.append(["run_ablation.py", "--biz", biz, "--variants", v, "--seeds", str(s),
                             "--tag", f"{biz}_v{i}_{s}"])
    return jobs


def decision_jobs(models, qmodels, quantiles):
    jobs = []
    for m in models:
        for s in C.SEEDS:
            jobs.append(["run_decision_forecasts.py", "--models", m, "--quantile-models",
                         "--seeds", str(s), "--tag", f"l1_{m}_{s}"])
    for m in qmodels:
        for q in quantiles:
            for s in C.SEEDS:
                jobs.append(["run_decision_forecasts.py", "--models", "--quantile-models", m,
                             "--quantiles", str(q), "--seeds", str(s), "--tag", f"pb_{m}_{q}_{s}"])
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["decision", "benchmark", "ablation"])
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--gpus", nargs="*", default=["0"])
    ap.add_argument("--threads", type=int, default=2, help="CPU threads per job")
    ap.add_argument("--neural-models", nargs="*", default=C.NEURAL_MODELS,
                    help="neural models for the benchmark and the L1 decision runs")
    ap.add_argument("--quantile-models", nargs="*", default=["AMSGCN", "AGCRN", "GraphWaveNet", "STAEformer"])
    ap.add_argument("--skip-det", action="store_true", help="statistical baselines already done")
    ap.add_argument("--quantiles", nargs="*", type=float, default=[0.6, 0.7, 0.8, 0.9])
    args = ap.parse_args()

    logs = C.OUT_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    # The deterministic decision forecasts also write the shared target arrays, so run them first.
    if "decision" in args.only and not args.skip_det:
        subprocess.run([PY, str(ROOT / "scripts" / "run_decision_forecasts.py"), "--models",
                        *C.DETERMINISTIC_MODELS, "--quantile-models", "--tag", "det"], check=True)

    jobs = []
    if "decision" in args.only:
        jobs += decision_jobs(args.neural_models, args.quantile_models, args.quantiles)
    if "benchmark" in args.only:
        jobs += benchmark_jobs(args.neural_models, not args.skip_det)
    if "ablation" in args.only:
        jobs += ablation_jobs()
    print(f"{len(jobs)} jobs, {args.jobs} at a time on GPUs {args.gpus}", flush=True)

    done = [0]
    t0 = time.time()

    def run(i_job):
        i, job = i_job
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus[i % len(args.gpus)],
                   OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
        tag = job[job.index("--tag") + 1]
        cmd = [PY, str(ROOT / "scripts" / job[0]), *job[1:], "--threads", str(args.threads)]
        with open(logs / f"{job[0].replace('.py', '')}_{tag}.log", "a") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        done[0] += 1
        print(f"[{done[0]}/{len(jobs)}] rc={rc} {job[0]} {tag}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        rcs = list(ex.map(run, enumerate(jobs)))
    bad = sum(r != 0 for r in rcs)
    print(f"finished: {len(jobs) - bad} ok, {bad} failed, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
