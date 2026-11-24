"""Spatiotemporal Transformer baseline in the style of STAEformer.

Reimplemented for ECE 592 from the description in Liu et al., "Spatio-Temporal
Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting"
(CIKM 2023). No code from the original repository is used.

Each input step of each node is embedded as the concatenation of
  - a linear projection of the observed demand,
  - a learned time-of-day embedding (96 slots of 15 minutes),
  - a learned spatiotemporal adaptive embedding indexed by (step, node),
and the resulting [B, T, N, D] tensor passes through vanilla Transformer
encoder layers that attend along the time axis (each node separately) and then
across nodes (each step separately). A mixed output projection maps the T x D
representation of every node to the PRED_LEN horizons.

Day-of-week is not used, for the reason given in amsgcn.PeriodicEmbedding: the
seven days in the sample are consecutive, so a day index is a date label that
the validation and test days never share with training.

The model has no positional encoding over nodes other than the node-indexed
adaptive embedding, so it is equivariant to a relabeling of the nodes
(checked in tests/test_node_axis.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .. import config as C


class STAEformer(nn.Module):
    def __init__(self, n_nodes, input_dim=24, tod_dim=24, adaptive_dim=32, heads=4,
                 ff_dim=128, temporal_layers=2, spatial_layers=2, dropout=0.1):
        super().__init__()
        self.model_dim = input_dim + tod_dim + adaptive_dim
        self.input_proj = nn.Linear(1, input_dim)
        self.tod = nn.Embedding(C.SLOTS_PER_DAY, tod_dim)
        self.adaptive = nn.Parameter(torch.empty(C.SEQ_LEN, n_nodes, adaptive_dim))
        nn.init.xavier_uniform_(self.adaptive)

        def layer():
            return nn.TransformerEncoderLayer(self.model_dim, heads, ff_dim, dropout, batch_first=True)

        self.temporal = nn.ModuleList([layer() for _ in range(temporal_layers)])
        self.spatial = nn.ModuleList([layer() for _ in range(spatial_layers)])
        self.output_proj = nn.Linear(C.SEQ_LEN * self.model_dim, C.PRED_LEN)

    def forward(self, batch):
        x = batch["x"]                                            # [B, T, N, 1]
        b, t, n, _ = x.shape
        tod = self.tod(batch["tod"][:, :t])                       # [B, T, d_tod]
        h = torch.cat([
            self.input_proj(x),
            tod.unsqueeze(2).expand(b, t, n, -1),
            self.adaptive.unsqueeze(0).expand(b, t, n, -1),
        ], dim=-1)                                                # [B, T, N, D]
        d = self.model_dim

        # attention along time, one sequence per (sample, node)
        h = h.permute(0, 2, 1, 3).reshape(b * n, t, d)
        for blk in self.temporal:
            h = blk(h)
        h = h.reshape(b, n, t, d)

        # attention across nodes, one set per (sample, step)
        h = h.permute(0, 2, 1, 3).reshape(b * t, n, d)
        for blk in self.spatial:
            h = blk(h)
        h = h.reshape(b, t, n, d)

        # mixed output projection: all steps of a node -> its PRED_LEN horizons
        h = h.permute(0, 2, 1, 3).reshape(b, n, t * d)
        y = self.output_proj(h)                                   # [B, N, P]
        return y.permute(0, 2, 1).unsqueeze(-1)                   # [B, P, N, 1]
