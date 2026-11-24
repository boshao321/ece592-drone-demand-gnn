# Decision-Aware Spatiotemporal Graph Forecasting for Drone Fleet Allocation

Course project for NCSU ECE 592 / CSC 591, Advanced Topics in Deep Learning (Fall 2026).
Author: Bo Shao (bshao3@ncsu.edu).

The project forecasts 15-minute delivery demand on a 120-cell grid of Nanshan District,
Shenzhen, with spatiotemporal graph neural networks and a spatiotemporal Transformer, and
scores every forecast by the drone fleet plan it induces at 38 landing sites. The proposed
graph model, AMSGCN, combines multi-scale temporal convolution, time-of-day modulation,
graph convolution over the origin–destination flows observed in each input window, and a
node-adaptive output head. The proposed training objective replaces the usual L1 loss with
a pinball (quantile) loss whose level is chosen on a validation day by downstream fleet
performance. The decision metric is **gap closed (GC)**, the share of the gap between a
static time-of-day plan and a perfect-foresight plan that a forecast recovers.

## Results at a glance

Test-day gap closed (%), 5 seeds, t+60 forecast, quantile chosen on the validation day.
Full tables are in [`results/RESULTS.md`](results/RESULTS.md).

| Model and training | M = 50 | M = 100 | M = 200 | M = 300 | Mean |
|---|---|---|---|---|---|
| Static plan scaled by k (no learning) | 0.0 | 0.0 | 13.6 | 33.6 | 11.8 |
| Graph WaveNet, pinball | −18.0 | −12.0 | 21.4 | 44.2 | 8.9 |
| STAEformer (Transformer), pinball | 0.0 | −7.9 | 22.2 | 47.4 | 15.4 |
| AGCRN, pinball | −0.8 | 6.1 | **35.9** | **55.4** | 24.1 |
| **AMSGCN, pinball** | **15.4** | **6.2** | 28.6 | 50.0 | **25.0** |

With L1 training every model recovers at most about 15% of the gap at any fleet size.
AMSGCN is the most accurate model at the 60-minute lead time on the instant retail and
group meal lines, and the only model whose gap closed is positive for every seed at every
fleet size.

## Repository layout

```
core/                     library code
  config.py               paths, split, training and experiment settings
  data.py                 order loading, node set, demand tensors, OD graphs, datasets
  engine.py               training loop with L1 or pinball loss, early stopping, checkpoints
  metrics.py              per-horizon MAE, RMSE, WAPE
  allocation.py           landing-site scenario, exact fleet allocation, evaluation
  runner.py               model factory and statistical forecasts
  models/
    amsgcn.py             AMSGCN (proposed model)
    baselines.py          LSTM, STGCN, DCRNN, Graph WaveNet, AGCRN
    transformer.py        STAEformer (spatiotemporal Transformer baseline)
    statistical.py        historical average, seasonal naive, ridge
scripts/                  experiment entry points
  run_benchmark.py        experiment (i): forecasting benchmark on three business lines
  run_ablation.py         experiment (ii): AMSGCN component ablation
  run_decision_forecasts.py  experiment (iii-a): high-value forecasts, L1 and pinball training
  run_decision_eval.py    experiment (iii-b): fleet allocation, gap closed, quantile selection
  launch_parallel.py      runs every experiment as independent jobs on one or more GPUs
  summarize.py            report tables and figures
tests/
  test_node_axis.py       node-permutation equivariance and ablation integrity tests
data/
  sites_kstar.csv         coordinates of the 38 landing sites (grid cell centers)
results/                  summary tables, figures and RESULTS.md from the final run
```

## What I implemented

Every file in this repository was written by me. No code was copied from public
repositories. The baselines are reimplemented from their papers, and each reimplementation
keeps the node axis explicit so that `tests/test_node_axis.py` can verify it.

The work falls into two groups.

**Written for this course project (Sep to Nov 2026).** `core/config.py`, `core/runner.py`,
`core/models/transformer.py`, the pinball loss, validation predictions and checkpointing in
`core/engine.py`, the leakage-free high-value threshold in `core/data.py`, and every script in
`scripts/`. These implement the decision-aware training, the quantile selection by downstream
gap closed, the scaled static control, the Transformer baseline, the parallel experiment
launcher, and all tables and figures in the report.

**Reused from my own earlier research code on the same data, unchanged except as noted.**
`core/data.py`, `core/metrics.py`, `core/allocation.py`, `core/models/amsgcn.py`,
`core/models/baselines.py`, `core/models/statistical.py` and `tests/test_node_axis.py`.
AMSGCN and the baseline reimplementations come from this group.

## File, class and function reference

### `core/config.py`
Central settings. `REPO_ROOT`, `OUT_ROOT` (outputs, override with `ECE592_OUT`), `RAW_CSV`
(order file, override with `ECE592_DATA`), `SITES_CSV`. Task constants: 15-minute slots,
`SEQ_LEN = 12` input slots, `PRED_LEN = 4` horizons (t+15 to t+60), whole-day split
(days 1–5 train, 6 validation, 7 test), `BIZ_LINES`, `GRID_STEP = 0.01` degrees. Training:
`SEEDS` (42 to 46), batch 32, Adam with learning rate 1e-3 and weight decay 1e-4, at most
150 epochs, patience 20, `HIDDEN = 32`. Model lists `DETERMINISTIC_MODELS`,
`NEURAL_MODELS`, `ABLATION_VARIANTS`. Decision stage: `FLEET_SIZES`, `QUANTILES`,
`HIGH_VALUE_QUANTILE = 0.75`, `DECISION_HORIZON = 3` (t+60).
- `seed_everything(seed)` seeds Python, NumPy and PyTorch and makes cuDNN deterministic.

### `core/data.py`
- `haversine_km(lat1, lon1, lat2, lon2)` great-circle distance in kilometers.
- `load_orders()` reads the order CSV, converts payment time to a 15-minute slot index over the seven days, and drops rows without coordinates.
- `high_value_threshold_of(df)` 75th percentile of order value over days 1–6 only, so the test day never shapes the definition of a high-value order (53.90 on this data).
- `snap(values, step)` rounds coordinates to the grid.
- `class Spatial` the node set. `n` is the node count, `ids(lat, lng)` maps coordinates to node indices (−1 outside the set), `distance_adjacency()` builds a thresholded Gaussian kernel on haversine distance.
- `build_spatial(df, step, include_dropoff)` one node set shared by every business line; with `include_dropoff=True` it is the union of pickup and drop-off cells (120 cells).
- `demand_tensor(df, sp, biz, high_value_only, high_value_threshold)` order originations per cell per slot, `[672, N]`.
- `od_flow_per_slot(df, sp, biz)` sparse origin–destination counts per slot.
- `window_flow_adjacency(flow, sp, n_windows)` for every input window, the row-normalized matrix of OD counts observed inside that window, `A = rownorm(O + Oᵀ + I)`. This is AMSGCN's dynamic graph.
- `class StandardScaler` z-score scaling with `transform` and `inverse`.
- `class WindowDataset` a PyTorch dataset of 12-slot input windows and 4-slot targets; `__getitem__` returns `x`, `y`, time-of-day `tod`, day index `dow` and the window's dynamic adjacency `adj`.
- `class Bundle` train, validation and test datasets plus scaler, static adjacency, node count, all-zero baseline errors and the raw series.
- `build_bundle(df, sp, biz, need_dynamic, high_value_only, high_value_threshold)` builds a `Bundle` for one business line (or all lines with `biz=None`), with the whole-day split and scaling fitted on the training days.

### `core/engine.py`
- `class PinballLoss(q)` quantile loss `max(q·e, (q−1)·e)` with `e = y − ŷ`; `q = 0.5` is half the L1 loss.
- `pinball_np(p, y, q)` the same loss in NumPy, used for early stopping.
- `make_loader(ds, shuffle)`, `_to_device(batch, device)` data loading helpers.
- `collect(model, loader, scaler, device)` runs a model over a loader and returns predictions and labels on the original scale.
- `train_model(model, bundle, seed, ..., loss="l1"|"pinball", q, ckpt_path)` trains with Adam, gradient clipping and learning-rate decay on plateau, early-stops on the validation value of the loss being optimized, restores the best weights, and returns test metrics, test predictions and labels, and validation predictions and labels. With `ckpt_path` it saves state every epoch and resumes from it.

### `core/metrics.py`
- `per_horizon(preds, labels)` MAE, RMSE and WAPE for each horizon and averaged over horizons.

### `core/allocation.py`
Operating constants: cruise speed 60 km/h, 4 minutes of service per sortie, 3 km service radius, 8 km maximum trip, 15-minute slots.
- `sortie_minutes(dist_km)` aircraft minutes per order, `2·d/v + t_service`.
- `tour_minutes(...)` closed tour cost from the site (not used in the reported results).
- `class Scenario` per-slot, per-site aircraft-minute requirements, realized orders and a value-sorted order book.
- `assign_grids_to_sites(coords, site_idx, radius_km)` assigns each cell to its nearest site within the service radius.
- `build_scenario(orders, coords, site_idx, slots, ...)` turns raw orders into a `Scenario` for the slots being scored.
- `allocate(estimate_minutes, fleet_size)` exact optimum of `max Σ_k min(15·x_k, d̂_k)` subject to `Σ_k x_k ≤ M`, solved greedily site by site.
- `evaluate(x, sc)` serves the realized orders with the allocated capacity, highest value first, and reports the served high-value share and other service measures.
- `policy_estimates(sc, demand_full, ...)` aircraft-minute estimates of the static (time-of-day mean of the training days), lagged and oracle policies.
- `nodes_to_sites(pred_nodes, node_to_site, n_sites)` aggregates a cell-level forecast to sites.

### `core/runner.py`
- `build_neural(name, bundle, **kwargs)` constructs any neural model by name.
- `deterministic_preds(name, x_raw, starts)` historical-average and seasonal-naive forecasts.
- `targets(x_raw, starts)` the 4-slot ground truth of every window.

### `core/models/amsgcn.py` (proposed model)
- `class MultiScaleTemporalAttention` three dilated causal convolutions along time (dilation 1, 2, 4) fused by a softmax attention over scales.
- `class PeriodicEmbedding` a learned time-of-day embedding (96 slots) that modulates the temporal state by FiLM, `h·(1+γ) + β`.
- `class DynamicFlowGraphConv` second-order graph convolution `W[H, AH, A²H]` on the window's OD adjacency.
- `class NodeAdaptiveHead` per-node output weights generated from a 10-dimensional node embedding.
- `class AMSGCN(n_nodes, ..., use_multiscale, use_dynamic_graph, use_periodic, use_heterogeneity)` the full model, 17,209 parameters on 120 nodes. Each `use_*` flag removes one component for the ablation.

### `core/models/baselines.py`
- Graph operators `sym_norm_laplacian`, `scaled_laplacian`, `random_walk`.
- Layers `ChebConv` (Chebyshev spectral convolution), `TemporalGatedConv` (gated temporal convolution), `DiffusionConv` (bidirectional diffusion convolution), `DCGRUCell` (diffusion convolutional GRU cell), `AGCN` (node-adaptive graph convolution).
- Models `LSTMBaseline` (shared LSTM per node), `STGCN` (Yu et al. 2018), `DCRNN` (Li et al. 2018, with a linear decoder), `GraphWaveNet` (Wu et al. 2019), `AGCRN` (Bai et al. 2020).

### `core/models/transformer.py`
- `class STAEformer(n_nodes, ...)` spatiotemporal Transformer after Liu et al. (2023). Input projection, time-of-day embedding and a learned (step, node) adaptive embedding are concatenated, passed through two temporal and two spatial Transformer encoder layers, and mapped to the four horizons by a mixed output projection. 239,988 parameters.

### `core/models/statistical.py`
- `class HistoricalAverage` mean demand per node and time of day over the training days.
- `class SeasonalNaive` demand at the same time on the previous day.
- `class RidgeLinear` ridge regression on lags (not used in the reported results).

### `scripts/run_benchmark.py`
Experiment (i). For each business line, trains and scores the statistical baselines once and every neural model once per seed. Options `--biz`, `--models`, `--seeds`, `--max-epochs`, `--threads`, `--tag` (output file suffix), `--ckpt-dir`. Writes `outputs/tables/benchmark_runs_<tag>.csv`; finished runs are skipped on a rerun.

### `scripts/run_ablation.py`
Experiment (ii). Trains AMSGCN with each component removed in turn (`config.ABLATION_VARIANTS`). Options as above plus `--variants`. Writes `ablation_runs_<tag>.csv`.

### `scripts/run_decision_forecasts.py`
Experiment (iii-a). Builds the high-value demand target pooled over all business lines, then trains every model in `--models` with the L1 loss and every model in `--quantile-models` with the pinball loss at each `--quantiles` level. Saves validation and test forecasts to `outputs/forecasts_hv/pred_<model>_q<qq>_s<seed>.npz` and the shared targets and threshold once. Helper `qtag(q)` formats the file tag.

### `scripts/run_decision_eval.py`
Experiment (iii-b). For every saved forecast, aggregates the t+60 prediction to sites, allocates fleets of 50, 100, 200 and 300 drones on the validation day and the test day, and computes gap closed against the static and oracle plans. Also scores the static plan scaled by k from 1.0 to 2.5 (`SCALE_GRID`), and selects q (or k) per fleet size on the validation day. Helpers `load_sites(coords)` and `prepare_orders()`. Writes `decision_runs.csv`, `decision_gc_by_q.csv`, `decision_quantile_selected.csv` and `decision_reference_policies.csv`.

### `scripts/launch_parallel.py`
Expands experiments into single-run jobs (`benchmark_jobs`, `ablation_jobs`, `decision_jobs`) with unique tags and runs them in parallel, assigning GPUs round-robin. Options `--only`, `--jobs`, `--gpus`, `--threads`, `--neural-models`, `--quantile-models`, `--quantiles`, `--skip-det`.

### `scripts/summarize.py`
Merges all run files and writes `table_benchmark.csv/.tex`, `table_benchmark_ranks.csv`, `table_ablation.csv`, `table_decision.csv` and the figures `fig_mae_by_horizon`, `fig_gc_vs_fleet`, `fig_gc_vs_quantile` (PDF and PNG). Helpers `ms` (mean ± s.d. text), `read_all` (merge run files), `save` (write a figure), and one function per experiment (`benchmark`, `ablation`, `decision`).

### `tests/test_node_axis.py`
Permutes the nodes of the input, the adjacency and every node-indexed parameter and checks that each neural model's output is permuted the same way (`equivariance`, `copy_permuted`, `permute_batch`, `make_batch`). Also checks that each AMSGCN component responds to its input and that each ablation removes what it names.

## Data

The order records come from Meituan and are confidential, so they are not in this
repository. The file is one week (seven consecutive days) of desensitized orders in
Nanshan District, 592,667 rows, with pickup and drop-off coordinates, payment time, order
value and business line. Place it at `data/orders.csv` or set `ECE592_DATA`.

## Running the experiments

```bash
pip install -r requirements.txt
export ECE592_DATA=/path/to/orders.csv          # or copy the file to data/orders.csv
export ECE592_OUT=./outputs                     # optional

python tests/test_node_axis.py                  # must report "all node-axis tests passed"

# everything at once: 296 runs, about 35 minutes on 2 x A100 with 32 parallel jobs
python scripts/launch_parallel.py --jobs 32 --gpus 0 1
python scripts/run_decision_eval.py
python scripts/summarize.py
```

Single experiments can also be run sequentially:

```bash
python scripts/run_benchmark.py                          # (i)   3 lines x 9 models
python scripts/run_ablation.py                           # (ii)  3 lines x 5 variants x 5 seeds
python scripts/run_decision_forecasts.py                 # (iii) L1 models + pinball models
python scripts/run_decision_eval.py
python scripts/summarize.py

# one model, one seed, one line
python scripts/run_benchmark.py --biz shangou --models AMSGCN --seeds 42 --tag demo
```

Outputs go to `outputs/tables/` and `outputs/figures/`. The tables and figures of the
final run are copied in `results/`.
