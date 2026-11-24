"""Configuration for the ECE 592 course project.

Adapted from the research pipeline (2026-4-23/09_Forecasting_Rebuilt/core/config.py).
Course-specific changes:
  - one node set for everything: the 0.01-degree union of pickup and drop-off
    cells (N = 120), so forecasts can be handed to the fleet allocation;
  - three business lines (health is excluded as too sparse);
  - three seeds, 150 epochs max;
  - fleet sizes and pinball quantile levels for the decision-aware experiment;
  - data paths are resolved relative to this repository or from environment
    variables, so the raw order file never has to live inside the repo.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]              # .../ECE592/code
OUT_ROOT = Path(os.environ.get("ECE592_OUT", REPO_ROOT / "outputs"))

# Raw order records are proprietary and are NOT part of the repository.
# Point ECE592_DATA at the CSV, or place it at data/orders.csv (git-ignored).
RAW_CSV = Path(os.environ.get("ECE592_DATA", REPO_ROOT / "data" / "orders.csv"))
SITES_CSV = REPO_ROOT / "data" / "sites_kstar.csv"            # K* = 38 pads from the siting analysis

# ---------------------------------------------------------------- task setup
RESOLUTION_MIN = 15
SLOTS_PER_DAY = 1440 // RESOLUTION_MIN          # 96
N_DAYS = 7
TOTAL_SLOTS = SLOTS_PER_DAY * N_DAYS            # 672

SEQ_LEN = 12                                    # past 3 hours
PRED_LEN = 4                                    # t+15, t+30, t+45, t+60
HORIZON_LABELS = ["t+15", "t+30", "t+45", "t+60"]

TRAIN_DAYS, VAL_DAYS, TEST_DAYS = 5, 1, 1       # whole-day chronological split

BIZ_LINES = ["waimai", "shangou", "pinhaofan"]
BIZ_LABELS = {"waimai": "Food delivery", "shangou": "Instant retail",
              "pinhaofan": "Group meals", "health": "Health"}

GRID_STEP = 0.010                               # ~1.1 km cells
INCLUDE_DROPOFF = True                          # union node set, N = 120

ADJ_SIGMA_KM = 1.5                              # static distance graph (Li et al. 2018)
ADJ_THRESHOLD = 0.1

# ---------------------------------------------------------------- training
SEEDS = [42, 43, 44, 45, 46]
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 150
PATIENCE = 20
GRAD_CLIP = 5.0
HIDDEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DETERMINISTIC_MODELS = ["HA", "SeasonalNaive"]
NEURAL_MODELS = ["LSTM", "STGCN", "DCRNN", "GraphWaveNet", "AGCRN", "STAEformer", "AMSGCN"]
ALL_MODELS = DETERMINISTIC_MODELS + NEURAL_MODELS

ABLATION_VARIANTS = {
    "AMSGCN (full)": {},
    "w/o multi-scale temporal": {"use_multiscale": False},
    "w/o dynamic OD graph": {"use_dynamic_graph": False},
    "w/o periodic modulation": {"use_periodic": False},
    "w/o node-adaptive head": {"use_heterogeneity": False},
}

# ---------------------------------------------------------------- decision stage
FLEET_SIZES = [50, 100, 200, 300]
QUANTILES = [0.5, 0.6, 0.7, 0.8, 0.9]           # q = 0.5 is the ordinary L1 (median) objective
HIGH_VALUE_QUANTILE = 0.75                       # high value = top quartile of order value, days 1-6
DECISION_HORIZON = 3                            # index of t+60, the lead time fleet moves need


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
