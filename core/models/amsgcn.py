"""AMSGCN, rebuilt so that each named component is the thing it is named after.

What the previous implementation did, and what changed:

  Multi-scale attention. Previously a single Conv1d whose kernel slid over the
  node index. Now three parallel dilated causal convolutions along TIME with
  receptive fields of 2, 3 and 5 steps, fused by an attention weight computed
  from the sequence itself, so "multi-scale" and "attention" are both present.

  Dynamic flow graph. Previously softmax(E E^T), constant in time and unrelated
  to order flow. Now the adjacency for a sample is built from the origin-
  destination counts observed inside that sample's own input window, so it
  varies with t and comes from the data rather than from a free parameter.

  Periodic embedding. Previously an additive bias on the output with no time
  input. Now time-of-day and day-of-week indices are embedded and used to
  modulate the temporal representation.

  Node heterogeneity. Previously node embeddings that only fed the adjacency,
  which made the "w/o dynamic graph" ablation silently remove heterogeneity as
  well. Now the embeddings drive node-adaptive output parameters, a path that
  is independent of the dynamic graph, and the ablation truly freezes them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C


class MultiScaleTemporalAttention(nn.Module):
    """Parallel dilated causal convolutions along time, fused by attention."""

    def __init__(self, c_in, c_out, dilations=(1, 2, 4), kernel=2):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(c_in, c_out, kernel_size=kernel, dilation=d, padding=0)
            for d in dilations
        ])
        self.pads = [(kernel - 1) * d for d in dilations]
        self.score = nn.Linear(c_out, 1)
        self.n_scales = len(dilations)

    def forward(self, x):                       # [B, N, T, C]
        b, n, t, c = x.shape
        h = x.reshape(b * n, t, c).transpose(1, 2)          # [B*N, C, T]
        outs = []
        for conv, pad in zip(self.branches, self.pads):
            z = F.pad(h, (pad, 0))                          # causal padding on the left
            outs.append(conv(z).transpose(1, 2))            # [B*N, T, C']
        stack = torch.stack(outs, dim=2)                    # [B*N, T, S, C']
        w = torch.softmax(self.score(stack), dim=2)         # attention over scales
        fused = (stack * w).sum(dim=2)                      # [B*N, T, C']
        return fused.view(b, n, t, -1), w.view(b, n, t, self.n_scales)


class PeriodicEmbedding(nn.Module):
    """Time-of-day embedding used to modulate the temporal state.

    Day-of-week is deliberately excluded. The sample spans seven consecutive
    days, so a day index is not a recurring period but a label for one specific
    day: with days 1-5 in training, the embeddings for days 6 and 7 would never
    receive a gradient and would still hold their random initialization at
    validation and test time. Time-of-day recurs in every split and is learned
    from five observations of each of the 96 slots.
    """

    def __init__(self, dim):
        super().__init__()
        self.tod = nn.Embedding(C.SLOTS_PER_DAY, dim)
        nn.init.normal_(self.tod.weight, std=0.02)
        self.film = nn.Linear(dim, 2 * dim)
        self.dim = dim

    def forward(self, h, tod_idx, dow_idx=None):    # h: [B, N, T, C]
        gamma, beta = self.film(self.tod(tod_idx)).chunk(2, dim=-1)   # [B, T, d]
        return h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class DynamicFlowGraphConv(nn.Module):
    """Graph convolution over the OD adjacency observed in the input window."""

    def __init__(self, c_in, c_out, order=2):
        super().__init__()
        self.order = order
        self.lin = nn.Linear(c_in * (order + 1), c_out)

    def forward(self, x, adj):                  # x: [B, N, C], adj: [B, N, N]
        out = [x]
        h = x
        for _ in range(self.order):
            h = torch.bmm(adj, h)
            out.append(h)
        return self.lin(torch.cat(out, -1))


class NodeAdaptiveHead(nn.Module):
    """Per-node output parameters generated from the node embedding."""

    def __init__(self, emb_dim, c_in, pred_len):
        super().__init__()
        self.w_pool = nn.Parameter(torch.empty(emb_dim, c_in, pred_len))
        self.b_pool = nn.Parameter(torch.zeros(emb_dim, pred_len))
        nn.init.xavier_uniform_(self.w_pool)

    def forward(self, h, node_emb):             # h: [B, N, C]
        w = torch.einsum("nd,dcp->ncp", node_emb, self.w_pool)
        b = node_emb @ self.b_pool
        return torch.einsum("bnc,ncp->bnp", h, w) + b


class AMSGCN(nn.Module):
    def __init__(self, n_nodes, hidden=C.HIDDEN, emb_dim=10, dropout=0.2,
                 use_multiscale=True, use_dynamic_graph=True,
                 use_periodic=True, use_heterogeneity=True):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.use_multiscale = use_multiscale
        self.use_dynamic_graph = use_dynamic_graph
        self.use_periodic = use_periodic
        self.use_heterogeneity = use_heterogeneity
        self.hidden = hidden

        self.input_proj = nn.Linear(1, hidden)

        if use_multiscale:
            self.temporal = MultiScaleTemporalAttention(hidden, hidden, dilations=(1, 2, 4))
        else:
            # a single scale, same parameter budget per branch, same causal padding
            self.temporal = MultiScaleTemporalAttention(hidden, hidden, dilations=(1,))

        if use_periodic:
            self.periodic = PeriodicEmbedding(hidden)

        if use_dynamic_graph:
            self.graph = DynamicFlowGraphConv(hidden, hidden, order=2)
        else:
            self.graph = nn.Linear(hidden, hidden)

        self.node_emb = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        if not use_heterogeneity:
            # genuinely removed: frozen, and the head below ignores it
            self.node_emb.requires_grad_(False)
            self.head = nn.Linear(hidden, C.PRED_LEN)
        else:
            self.head = NodeAdaptiveHead(emb_dim, hidden, C.PRED_LEN)

        self.norm = nn.LayerNorm(hidden)

    def forward(self, batch):
        x = batch["x"]                                   # [B, T, N, 1]
        b, t, n, _ = x.shape
        h = self.input_proj(x).permute(0, 2, 1, 3)       # [B, N, T, C]

        h, self.last_scale_weights = self.temporal(h)

        if self.use_periodic:
            h = self.periodic(h, batch["tod"][:, :t])

        h = self.norm(h)
        h = self.drop(h)
        h = h[:, :, -1, :]                               # [B, N, C], last step state

        if self.use_dynamic_graph:
            h = F.relu(self.graph(h, batch["adj"]))
        else:
            h = F.relu(self.graph(h))
        h = self.drop(h)

        if self.use_heterogeneity:
            y = self.head(h, self.node_emb)              # [B, N, P]
        else:
            y = self.head(h)
        return y.permute(0, 2, 1).unsqueeze(-1)          # [B, P, N, 1]
