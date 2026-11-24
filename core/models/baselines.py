"""Faithful baseline implementations.

Every model here keeps the node axis explicit from input to output. The audit
of the previous code found a reshape that collapsed [B, C, T, N] straight to
[B, N, C*T] without moving N first, which silently mixed nodes; the permutation
test in tests/test_node_axis.py guards against that class of bug.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C


# --------------------------------------------------------------- graph utils


def sym_norm_laplacian(w: np.ndarray) -> np.ndarray:
    d = w.sum(1)
    d_inv_sqrt = np.zeros_like(d)
    np.power(d, -0.5, out=d_inv_sqrt, where=d > 0)
    norm_adj = d_inv_sqrt[:, None] * w * d_inv_sqrt[None, :]
    return (np.eye(len(w), dtype=np.float32) - norm_adj).astype(np.float32)


def scaled_laplacian(w: np.ndarray) -> np.ndarray:
    lap = sym_norm_laplacian(w)
    eig = np.linalg.eigvalsh(lap)
    lam = float(eig.max()) if len(eig) else 2.0
    lam = lam if lam > 0 else 2.0
    return (2.0 * lap / lam - np.eye(len(w), dtype=np.float32)).astype(np.float32)


def random_walk(w: np.ndarray) -> np.ndarray:
    d = w.sum(1, keepdims=True)
    return (w / np.where(d > 0, d, 1.0)).astype(np.float32)


# ------------------------------------------------------------ neural modules


class ChebConv(nn.Module):
    """Chebyshev spectral graph convolution (Defferrard et al., used by STGCN)."""

    def __init__(self, c_in, c_out, k, l_tilde):
        super().__init__()
        self.k = k
        self.register_buffer("l_tilde", torch.from_numpy(l_tilde))
        self.theta = nn.Parameter(torch.empty(k, c_in, c_out))
        nn.init.xavier_uniform_(self.theta)
        self.bias = nn.Parameter(torch.zeros(c_out))

    def forward(self, x):                       # x: [B, T, N, C]
        xs = [x]
        if self.k > 1:
            xs.append(torch.einsum("nm,btmc->btnc", self.l_tilde, x))
        for _ in range(2, self.k):
            xs.append(2 * torch.einsum("nm,btmc->btnc", self.l_tilde, xs[-1]) - xs[-2])
        out = sum(torch.einsum("btnc,co->btno", xs[i], self.theta[i]) for i in range(self.k))
        return out + self.bias


class TemporalGatedConv(nn.Module):
    """Gated causal convolution along TIME (STGCN's temporal block)."""

    def __init__(self, c_in, c_out, kt=3):
        super().__init__()
        self.kt = kt
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(kt, 1))
        self.res = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else nn.Identity()

    def forward(self, x):                       # [B, T, N, C]
        h = x.permute(0, 3, 1, 2)               # [B, C, T, N]
        r = self.res(h)[:, :, self.kt - 1:, :]
        h = self.conv(h)
        p, q = h.chunk(2, dim=1)
        h = (p + r) * torch.sigmoid(q)
        return h.permute(0, 2, 3, 1)            # [B, T', N, C]


class DiffusionConv(nn.Module):
    """Diffusion convolution over forward and backward random walks (DCRNN)."""

    def __init__(self, c_in, c_out, k, supports):
        super().__init__()
        self.k = k
        for i, s in enumerate(supports):
            self.register_buffer(f"sup{i}", torch.from_numpy(s))
        self.n_sup = len(supports)
        self.lin = nn.Linear(c_in * (1 + self.n_sup * k), c_out)

    def forward(self, x):                       # [B, N, C]
        out = [x]
        for i in range(self.n_sup):
            s = getattr(self, f"sup{i}")
            h = x
            for _ in range(self.k):
                h = torch.einsum("nm,bmc->bnc", s, h)
                out.append(h)
        return self.lin(torch.cat(out, dim=-1))


class DCGRUCell(nn.Module):
    def __init__(self, c_in, hidden, k, supports):
        super().__init__()
        self.hidden = hidden
        self.gate = DiffusionConv(c_in + hidden, 2 * hidden, k, supports)
        self.cand = DiffusionConv(c_in + hidden, hidden, k, supports)

    def forward(self, x, h):
        z_r = torch.sigmoid(self.gate(torch.cat([x, h], -1)))
        z, r = z_r.chunk(2, dim=-1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], -1)))
        return z * h + (1 - z) * c


class AGCN(nn.Module):
    """Node-adaptive graph convolution with per-node parameters (AGCRN)."""

    def __init__(self, c_in, c_out, emb_dim, cheb_k=2):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.empty(emb_dim, cheb_k, c_in, c_out))
        self.bias_pool = nn.Parameter(torch.empty(emb_dim, c_out))
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias_pool)

    def forward(self, x, node_emb):             # x: [B, N, C]
        n = node_emb.shape[0]
        supports = F.softmax(F.relu(node_emb @ node_emb.T), dim=1)
        sup = [torch.eye(n, device=x.device), supports]
        for _ in range(2, self.cheb_k):
            sup.append(2 * supports @ sup[-1] - sup[-2])
        sup = torch.stack(sup[: self.cheb_k], 0)                 # [K, N, N]
        w = torch.einsum("nd,dkio->nkio", node_emb, self.weights)
        b = node_emb @ self.bias_pool
        g = torch.einsum("knm,bmc->bknc", sup, x)
        return torch.einsum("bknc,nkco->bno", g, w) + b


# ----------------------------------------------------------------- baselines


class LSTMBaseline(nn.Module):
    """Shared LSTM applied independently to each node's own series."""

    def __init__(self, n_nodes, hidden=C.HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, C.PRED_LEN)

    def forward(self, batch):
        x = batch["x"]                                   # [B, T, N, 1]
        b, t, n, _ = x.shape
        h = x.permute(0, 2, 1, 3).reshape(b * n, t, 1)   # node axis moved out first
        out, _ = self.lstm(h)
        y = self.head(out[:, -1])                        # [B*N, P]
        return y.view(b, n, C.PRED_LEN).permute(0, 2, 1).unsqueeze(-1)


class STGCN(nn.Module):
    """Spatio-temporal graph convolution: gated temporal conv, Chebyshev graph
    conv, gated temporal conv, then an output layer."""

    def __init__(self, n_nodes, adj, hidden=C.HIDDEN, cheb_k=3):
        super().__init__()
        l_tilde = scaled_laplacian(adj)
        self.t1 = TemporalGatedConv(1, hidden, kt=3)
        self.g1 = ChebConv(hidden, hidden, cheb_k, l_tilde)
        self.t2 = TemporalGatedConv(hidden, hidden, kt=3)
        self.norm = nn.LayerNorm(hidden)
        remaining = C.SEQ_LEN - 2 * (3 - 1)
        self.head = nn.Linear(hidden * remaining, C.PRED_LEN)

    def forward(self, batch):
        h = self.t1(batch["x"])
        h = F.relu(self.g1(h))
        h = self.t2(h)
        h = self.norm(h)
        b, t, n, c = h.shape
        h = h.permute(0, 2, 1, 3).reshape(b, n, t * c)   # node axis first, then flatten
        y = self.head(h)
        return y.permute(0, 2, 1).unsqueeze(-1)


class DCRNN(nn.Module):
    """Diffusion convolutional recurrent encoder with a linear decoder.

    The published model uses a recurrent decoder with scheduled sampling; with
    465 training windows that decoder overfits badly, so the encoder state is
    projected to all horizons at once. The diffusion convolution itself is
    unchanged.
    """

    def __init__(self, n_nodes, adj, hidden=C.HIDDEN, k=2):
        super().__init__()
        supports = [random_walk(adj), random_walk(adj.T)]
        self.cell1 = DCGRUCell(1, hidden, k, supports)
        self.cell2 = DCGRUCell(hidden, hidden, k, supports)
        self.hidden = hidden
        self.head = nn.Linear(hidden, C.PRED_LEN)

    def forward(self, batch):
        x = batch["x"]
        b, t, n, _ = x.shape
        h1 = x.new_zeros(b, n, self.hidden)
        h2 = x.new_zeros(b, n, self.hidden)
        for i in range(t):
            h1 = self.cell1(x[:, i], h1)
            h2 = self.cell2(h1, h2)
        y = self.head(h2)                                # [B, N, P]
        return y.permute(0, 2, 1).unsqueeze(-1)


class GraphWaveNet(nn.Module):
    """Dilated causal TCN with gated activations, a fixed diffusion support and
    a learned adaptive adjacency."""

    def __init__(self, n_nodes, adj, hidden=C.HIDDEN, emb_dim=10, layers=(1, 2, 4)):
        super().__init__()
        self.start = nn.Conv2d(1, hidden, (1, 1))
        self.filters = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.gconvs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for d in layers:
            self.filters.append(nn.Conv2d(hidden, hidden, (2, 1), dilation=(d, 1)))
            self.gates.append(nn.Conv2d(hidden, hidden, (2, 1), dilation=(d, 1)))
            self.gconvs.append(DiffusionConv(hidden, hidden, 2, [random_walk(adj), random_walk(adj.T)]))
            self.bns.append(nn.BatchNorm2d(hidden))
        self.e1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.e2 = nn.Parameter(torch.randn(emb_dim, n_nodes) * 0.1)
        self.adapt_mix = nn.Linear(hidden, hidden)
        self.end1 = nn.Conv2d(hidden, hidden, (1, 1))
        self.end2 = nn.Conv2d(hidden, C.PRED_LEN, (1, 1))

    def forward(self, batch):
        x = batch["x"].permute(0, 3, 1, 2)               # [B, 1, T, N]
        h = self.start(x)
        skip = 0
        adp = F.softmax(F.relu(self.e1 @ self.e2), dim=1)
        for filt, gate, gconv, bn in zip(self.filters, self.gates, self.gconvs, self.bns):
            res = h
            f = torch.tanh(filt(h))
            g = torch.sigmoid(gate(h))
            h = f * g
            steps = h.shape[2]
            hs = h.permute(0, 2, 3, 1).reshape(-1, h.shape[3], h.shape[1])   # [B*T, N, C]
            hs = gconv(hs) + self.adapt_mix(torch.einsum("nm,bmc->bnc", adp, hs))
            h = hs.view(h.shape[0], steps, h.shape[3], h.shape[1]).permute(0, 3, 1, 2)
            h = bn(h)
            h = h + res[:, :, -h.shape[2]:, :]
            skip = h[:, :, -1:, :] + (skip if isinstance(skip, int) else skip[:, :, -1:, :])
        h = F.relu(self.end1(F.relu(skip)))
        y = self.end2(h)                                  # [B, P, 1, N]
        return y.squeeze(2).unsqueeze(-1)


class AGCRN(nn.Module):
    """Adaptive graph convolutional recurrent network with node-adaptive
    parameter learning."""

    def __init__(self, n_nodes, hidden=C.HIDDEN, emb_dim=10):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.gate = AGCN(1 + hidden, 2 * hidden, emb_dim)
        self.cand = AGCN(1 + hidden, hidden, emb_dim)
        self.hidden = hidden
        self.head = nn.Linear(hidden, C.PRED_LEN)

    def forward(self, batch):
        x = batch["x"]
        b, t, n, _ = x.shape
        h = x.new_zeros(b, n, self.hidden)
        for i in range(t):
            inp = torch.cat([x[:, i], h], -1)
            z, r = torch.sigmoid(self.gate(inp, self.node_emb)).chunk(2, -1)
            c = torch.tanh(self.cand(torch.cat([x[:, i], r * h], -1), self.node_emb))
            h = z * h + (1 - z) * c
        y = self.head(h)
        return y.permute(0, 2, 1).unsqueeze(-1)
