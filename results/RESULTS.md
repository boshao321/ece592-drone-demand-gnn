# Results (final run, 2026-11-20, NCSU VCL, 2 x A100)

Changes from the September run: five seeds for every
neural model, the high-value threshold computed from days 1-6 only (53.90 instead of
54.80 from all seven days), and pinball training extended to Graph WaveNet and to
q = 0.9 for every pinball model, and a spatiotemporal Transformer baseline (STAEformer,
reimplemented) added on 2026-11-21. 296 runs in total. Test day = day 7, validation
day = day 6. Neural models are mean ± s.d. over 5 seeds.

## (i) Forecasting benchmark, 120 cells, test MAE (orders per cell per 15 min)

| Model | Food delivery MAE | MAE@t+60 | Instant retail MAE | MAE@t+60 | Group meals MAE | MAE@t+60 |
|---|---|---|---|---|---|---|
| All zero | 6.043 | 6.054 | 0.685 | 0.686 | 0.317 | 0.318 |
| Historical avg. | 2.173 | 2.172 | 0.403 | 0.403 | 0.274 | 0.274 |
| Seasonal naive | 1.580 | **1.588** | 0.461 | 0.461 | 0.237 | 0.237 |
| LSTM | 2.007 ± 0.012 | 2.519 ± 0.018 | 0.394 ± 0.001 | 0.408 ± 0.001 | 0.229 ± 0.001 | 0.247 ± 0.001 |
| STGCN | 1.792 ± 0.038 | 2.194 ± 0.062 | 0.384 ± 0.003 | 0.395 ± 0.003 | 0.223 ± 0.001 | 0.243 ± 0.001 |
| DCRNN | 1.765 ± 0.039 | 2.096 ± 0.087 | 0.389 ± 0.006 | 0.400 ± 0.006 | 0.224 ± 0.002 | 0.242 ± 0.002 |
| Graph WaveNet | **1.566 ± 0.020** | 1.748 ± 0.043 | 0.366 ± 0.004 | 0.373 ± 0.004 | 0.210 ± 0.002 | 0.218 ± 0.003 |
| AGCRN | 1.623 ± 0.040 | 1.816 ± 0.060 | 0.364 ± 0.001 | 0.371 ± 0.001 | 0.216 ± 0.005 | 0.231 ± 0.005 |
| STAEformer (Transformer) | 1.912 ± 0.053 | 1.998 ± 0.079 | 0.366 ± 0.002 | 0.366 ± 0.002 | 0.219 ± 0.005 | 0.218 ± 0.006 |
| AMSGCN | 1.858 ± 0.024 | 1.941 ± 0.060 | **0.361 ± 0.001** | **0.363 ± 0.001** | **0.209 ± 0.003** | **0.203 ± 0.007** |

AMSGCN ranks first on instant retail and group meals, both over all horizons and at
t+60. On food delivery it ranks 6th overall and 4th at t+60. STAEformer ranks 4th, 4th
and 7th overall and 2nd, 3rd and 5th at t+60 (instant retail, group meals, food delivery).

## (ii) AMSGCN ablation, change in test MAE@t+60 when a component is removed

| Removed | Food delivery | Instant retail | Group meals |
|---|---|---|---|
| Node-adaptive head | +10.9% | +6.8% | +10.7% |
| Periodic modulation | +3.2% | +3.7% | +9.1% |
| Dynamic OD graph | -2.4% | +0.4% | -2.1% |
| Multi-scale temporal conv | -2.7% | +0.4% | +0.5% |

## (iii) Fleet allocation, 38 sites, t+60 forecast, gap closed on the test day (%)

Served high-value share, static -> oracle: M=50 25.9 -> 28.6, M=100 45.6 -> 51.4,
M=200 68.0 -> 79.8, M=300 76.8 -> 94.2.

| Method | M=50 | M=100 | M=200 | M=300 | Mean |
|---|---|---|---|---|---|
| Historical avg. | -0.5 | -0.1 | 0.1 | 0.0 | |
| Seasonal naive | -77.8 | -47.1 | -12.3 | 5.8 | |
| LSTM (L1) | -21.3 ± 8.6 | -38.4 ± 5.0 | -52.3 ± 2.3 | -58.7 ± 1.5 | |
| STGCN (L1) | -11.7 ± 5.3 | -9.7 ± 4.5 | -24.7 ± 2.5 | -35.9 ± 3.2 | |
| DCRNN (L1) | -10.2 ± 3.8 | -12.0 ± 2.4 | -14.3 ± 4.3 | -17.2 ± 6.9 | |
| Graph WaveNet (L1) | -18.0 ± 10.4 | -12.0 ± 7.1 | 1.4 ± 3.9 | 6.4 ± 10.2 | |
| AGCRN (L1) | -0.8 ± 4.4 | 6.1 ± 2.6 | 10.2 ± 8.5 | 1.1 ± 8.8 | |
| STAEformer (L1) | -0.0 ± 5.3 | -7.9 ± 2.8 | 4.7 ± 1.8 | 10.9 ± 1.6 | |
| AMSGCN (L1) | **15.4 ± 2.4** | **6.7 ± 3.0** | 5.4 ± 4.3 | **7.7 ± 2.9** | |
| Static x k, k from val. | 0.0 | 0.0 | 13.6 | 33.6 | 11.8 |
| Graph WaveNet, pinball, q from val. | -18.0 ± 10.4 | -12.0 ± 7.1 | 21.4 ± 5.1 | 44.2 ± 2.6 | 8.9 |
| STAEformer, pinball, q from val. | -0.0 ± 5.3 | -7.9 ± 2.8 | 22.2 ± 2.0 | 47.4 ± 2.7 | 15.4 |
| AGCRN, pinball, q from val. | -0.8 ± 4.4 | 6.1 ± 2.6 | **35.9 ± 3.9** | **55.4 ± 1.4** | 24.1 |
| AMSGCN, pinball, q from val. | **15.4 ± 2.4** | 6.2 ± 1.9 | 28.6 ± 2.6 | 50.0 ± 2.9 | **25.0** |

Selected q: 0.5, 0.5 or 0.6, 0.7 and 0.8 for M = 50, 100, 200, 300, the same for all
four models. Pinball training lifts every neural model far above both its L1
version and the no-learning scaled static plan at M = 200 and 300. AMSGCN has the
highest mean gap closed across fleet sizes and is the only model whose gap closed is
positive for every seed at every fleet size; AGCRN leads at the two largest fleets.

## Caveats

One validation day and one test day; seed s.d. does not capture day-to-day variation.
DCRNN, AGCRN and Graph WaveNet often reach the 150-epoch cap rather than early stopping.
