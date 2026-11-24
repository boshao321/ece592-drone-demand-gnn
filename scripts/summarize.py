"""Tables and figures for the report (written for ECE 592).

Reads the raw run files in outputs/tables and writes
  outputs/tables/table_benchmark.csv / .tex   per-horizon accuracy, mean +/- s.d.
  outputs/tables/table_ablation.csv / .tex
  outputs/tables/table_decision.csv / .tex    test-day GC by fleet size
  outputs/figures/fig_gc_vs_fleet.pdf/.png
  outputs/figures/fig_mae_by_horizon.pdf/.png
  outputs/figures/fig_gc_vs_quantile.pdf/.png

    python scripts/summarize.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config as C             # noqa: E402

TABLES = C.OUT_ROOT / "tables"
FIGS = C.OUT_ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# categorical slots in fixed order (validated reference palette, light mode)
COL = {"AMSGCN": "#2a78d6", "AGCRN": "#eb6834", "GraphWaveNet": "#4a3aa7",
       "SeasonalNaive": "#eda100", "HA": "#8a8985", "STGCN": "#e87ba4",
       "DCRNN": "#008300", "LSTM": "#e34948", "STAEformer": "#1baf7a", "ZeroBaseline": "#8a8985", "StaticScaled": "#52514e"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["ZeroBaseline", "HA", "SeasonalNaive", "LSTM", "STGCN", "DCRNN", "GraphWaveNet", "AGCRN", "STAEformer", "AMSGCN"]
LABEL = {"ZeroBaseline": "All zero", "SeasonalNaive": "Seasonal naive", "GraphWaveNet": "Graph WaveNet",
         "HA": "Historical avg.", "StaticScaled": r"Static plan $\times k$", "AMSGCN": "AMSGCN (ours)"}

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "legend.frameon": False, "lines.linewidth": 1.6})


def ms(mean, std, digits=3):
    return f"{mean:.{digits}f}" if not np.isfinite(std) else f"{mean:.{digits}f} ± {std:.{digits}f}"


def read_all(prefix):
    fs = sorted(TABLES.glob(f"{prefix}_*.csv"))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    keys = [c for c in ("biz", "model", "variant", "seed") if c in df.columns]
    return df.drop_duplicates(subset=keys, keep="first")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ benchmark
def benchmark():
    runs = read_all("benchmark_runs")
    if runs.empty:
        return
    cols = ["MAE@t+15", "MAE@t+60", "MAE", "RMSE", "WAPE"]
    g = runs.groupby(["biz", "model"])[cols]
    mean, std, n = g.mean(), g.std(), g.size()
    rows = []
    for biz in C.BIZ_LINES:
        for m in ORDER:
            if (biz, m) not in mean.index:
                continue
            r = {"biz": biz, "model": m, "n_seeds": int(n[(biz, m)])}
            for c in cols:
                r[c] = ms(mean.loc[(biz, m), c], std.loc[(biz, m), c], 1 if c == "WAPE" else 3)
                r[c + "_mean"] = mean.loc[(biz, m), c]
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "table_benchmark.csv", index=False)
    # rank of each model by mean MAE@t+60 within each line (all-zero excluded)
    t2 = t[t.model != "ZeroBaseline"].copy()
    t2["rank_t60"] = t2.groupby("biz")["MAE@t+60_mean"].rank()
    t2["rank_all"] = t2.groupby("biz")["MAE_mean"].rank()
    t2[["biz", "model", "rank_all", "rank_t60"]].to_csv(TABLES / "table_benchmark_ranks.csv", index=False)
    with open(TABLES / "table_benchmark.tex", "w") as f:
        for biz in C.BIZ_LINES:
            f.write(f"% {biz}\n")
            for _, r in t[t.biz == biz].iterrows():
                f.write(f"{LABEL.get(r.model, r.model)} & " + " & ".join(r[c] for c in cols) + " \\\\\n")

    # figure: MAE by horizon, one panel per line
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35))
    hz = ["MAE@t+15", "MAE@t+30", "MAE@t+45", "MAE@t+60"]
    show = ["SeasonalNaive", "GraphWaveNet", "AGCRN", "STAEformer", "AMSGCN"]
    for ax, biz in zip(axes, C.BIZ_LINES):
        sub = runs[runs.biz == biz]
        for m in show:
            s = sub[sub.model == m]
            if s.empty:
                continue
            y = s[hz].mean().to_numpy()
            ax.plot([15, 30, 45, 60], y, marker="o", ms=3.5, color=COL[m], label=LABEL.get(m, m))
        ax.set_title(C.BIZ_LABELS[biz], fontsize=8, color=INK)
        ax.set_xticks([15, 30, 45, 60])
        ax.set_xlabel("Lead time (min)")
    axes[0].set_ylabel("Test MAE (orders per cell)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "fig_mae_by_horizon")


# ------------------------------------------------------------------ ablation
def ablation():
    runs = read_all("ablation_runs")
    if runs.empty:
        return
    cols = ["MAE", "MAE@t+60"]
    g = runs.groupby(["biz", "variant"])[cols]
    mean, std = g.mean(), g.std()
    rows = []
    for biz in C.BIZ_LINES:
        if (biz, "AMSGCN (full)") not in mean.index:
            continue
        full = mean.loc[(biz, "AMSGCN (full)")]
        for v in C.ABLATION_VARIANTS:
            if (biz, v) not in mean.index:
                continue
            r = {"biz": biz, "variant": v}
            for c in cols:
                r[c] = ms(mean.loc[(biz, v), c], std.loc[(biz, v), c])
                r[c + "_change_pct"] = 100 * (mean.loc[(biz, v), c] / full[c] - 1)
            rows.append(r)
    pd.DataFrame(rows).to_csv(TABLES / "table_ablation.csv", index=False)


# ------------------------------------------------------------------ decision
def decision():
    p = TABLES / "decision_gc_by_q.csv"
    if not p.exists():
        return
    summ = pd.read_csv(p)
    sel = pd.read_csv(TABLES / "decision_quantile_selected.csv") if (TABLES / "decision_quantile_selected.csv").exists() else pd.DataFrame()
    test = summ[summ.split == "test"]
    base = test[test.loss.isin(["l1", "det"])]
    rows = []
    for m in ORDER:
        s = base[base.model == m]
        if s.empty:
            continue
        r = {"method": f"{LABEL.get(m, m)}" + (" (L1)" if s.loss.iloc[0] == "l1" else "")}
        for _, x in s.sort_values("fleet").iterrows():
            r[f"M={int(x.fleet)}"] = ms(x["mean"], x["std"], 1)
        rows.append(r)
    for m in sel.model.unique() if len(sel) else []:
        s = sel[sel.model == m].sort_values("fleet")
        knob = "k" if m == "StaticScaled" else "q"
        r = {"method": f"{LABEL.get(m, m)} ({'k' if m == 'StaticScaled' else 'pinball, q'} from val.)"}
        for _, x in s.iterrows():
            r[f"M={int(x.fleet)}"] = ms(x.test_GC_selected_mean, x.test_GC_selected_std, 1) + f" [{knob}={x.q_selected:.1f}]"
        rows.append(r)
    pd.DataFrame(rows).to_csv(TABLES / "table_decision.csv", index=False)

    # figure: GC vs fleet size on the test day (the full model list is in the table)
    fig, ax = plt.subplots(figsize=(5.4, 2.55))
    ax.axhline(0, color=MUTED, lw=0.8)
    fleets = C.FLEET_SIZES
    for m in ["GraphWaveNet", "AGCRN", "STAEformer", "AMSGCN"]:
        s = base[base.model == m].sort_values("fleet")
        ax.plot(s.fleet, s["mean"], ls="--", marker="o", ms=2.5, lw=1.0, color=COL[m], alpha=0.85,
                label=f"{LABEL.get(m, m)}, L1")
    for m in ["StaticScaled", "GraphWaveNet", "STAEformer", "AGCRN", "AMSGCN"]:
        s = sel[sel.model == m].sort_values("fleet") if len(sel) else pd.DataFrame()
        if s.empty:
            continue
        if m == "StaticScaled":
            ax.plot(s.fleet, s.test_GC_selected_mean, color=COL[m], ls=":", marker="^", ms=4, lw=1.4,
                    label=r"Static plan $\times k$, no learning")
        else:
            ax.errorbar(s.fleet, s.test_GC_selected_mean, yerr=s.test_GC_selected_std, color=COL[m],
                        marker="s", ms=3.5, capsize=2, lw=1.8, label=f"{LABEL.get(m, m)}, pinball")
    ax.set_xticks(fleets)
    ax.set_xlim(35, 320)
    ax.set_xlabel("Fleet size $M$ (drones)")
    ax.set_ylabel("Gap closed, test day (%)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    save(fig, "fig_gc_vs_fleet")

    # figure: validation GC vs q for the pinball-trained models
    val = summ[(summ.split == "val") & summ.loss.isin(["l1", "pinball"])]
    ms_ = [m for m in ["GraphWaveNet", "AGCRN", "STAEformer", "AMSGCN"] if (val.model == m).sum() > len(fleets)]
    if ms_:
        fig, axes = plt.subplots(1, len(fleets), figsize=(6.6, 2.2), sharey=False)
        for ax, M in zip(axes, fleets):
            for m in ms_:
                s = val[(val.model == m) & (val.fleet == M)].sort_values("q")
                ax.plot(s.q, s["mean"], marker="o", ms=2.5, color=COL[m], label=LABEL.get(m, m))
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_title(f"$M = {M}$", fontsize=8)
            ax.set_xlabel("Pinball level $q$")
            ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        axes[0].set_ylabel("Gap closed, val. day (%)")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        save(fig, "fig_gc_vs_quantile")


if __name__ == "__main__":
    benchmark()
    ablation()
    decision()
    print("tables in", TABLES, "\nfigures in", FIGS)
