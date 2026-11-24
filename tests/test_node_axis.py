"""Guard tests for the bug class found in the audit of the previous code.

A model that treats nodes as a spatial axis must be equivariant to a relabeling
of the nodes: permute the input, the adjacency and every node-indexed parameter,
and the output must come back permuted the same way. The previous STGCN failed
this because it flattened [B, C, T, N] to [B, N, C*T] without moving N first,
which mixed different nodes into each output row.

The node count here is deliberately not equal to the embedding dimension, so
that a parameter's node axis can be identified unambiguously.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import config as C                      # noqa: E402
from core.models import baselines as B            # noqa: E402
from core.models.amsgcn import AMSGCN             # noqa: E402
from core.models.transformer import STAEformer    # noqa: E402

N = 13                  # != emb_dim (10), so the node axis is unambiguous
BS = 2
EMB = 10

rng = np.random.default_rng(0)
adj = rng.random((N, N)).astype(np.float32)
adj = (adj + adj.T) / 2
np.fill_diagonal(adj, 1.0)


def make_batch():
    torch.manual_seed(7)
    a = torch.rand(BS, N, N)
    a = a / a.sum(-1, keepdim=True)
    return {
        "x": torch.randn(BS, C.SEQ_LEN, N, 1),
        "y": torch.randn(BS, C.PRED_LEN, N, 1),
        "tod": torch.randint(0, C.SLOTS_PER_DAY, (BS, C.SEQ_LEN + C.PRED_LEN)),
        "dow": torch.randint(0, 7, (BS, C.SEQ_LEN + C.PRED_LEN)),
        "adj": a,
    }


def permute_batch(bt, perm):
    out = dict(bt)
    out["x"] = bt["x"][:, :, perm, :]
    out["y"] = bt["y"][:, :, perm, :]
    out["adj"] = bt["adj"][:, perm][:, :, perm]
    return out


def copy_permuted(m1, m2, perm):
    """Copy m1's parameters into m2, permuting any axis whose length is N.

    Buffers are left alone on purpose: m2 was constructed from the permuted
    adjacency, so its graph buffers already hold the permuted operator, and
    copying m1's would put the unpermuted graph back.
    """
    with torch.no_grad():
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            v = p1.detach().clone()
            for axis, size in enumerate(v.shape):
                if size == N:
                    v = v.index_select(axis, perm)
            p2.copy_(v)


def equivariance(name, ctor_from_adj, bt, perm, needs_adj):
    torch.manual_seed(1)
    m1 = ctor_from_adj(adj) if needs_adj else ctor_from_adj(None)
    torch.manual_seed(1)
    permuted_adj = adj[np.ix_(perm.numpy(), perm.numpy())]
    m2 = ctor_from_adj(permuted_adj) if needs_adj else ctor_from_adj(None)
    copy_permuted(m1, m2, perm)
    m1.eval(); m2.eval()
    with torch.no_grad():
        y1 = m1(bt)
        y2 = m2(permute_batch(bt, perm))
    diff = float((y2 - y1[:, :, perm, :]).abs().max())
    ok = diff < 1e-4
    print(f"  {name:<14} {'PASS' if ok else 'FAIL'}   max|diff| = {diff:.2e}")
    return ok


def main():
    bt = make_batch()
    torch.manual_seed(3)
    perm = torch.randperm(N)

    print("Node-axis equivariance (input, adjacency and node-indexed parameters permuted together):")
    results = {
        "LSTM": equivariance("LSTM", lambda a: B.LSTMBaseline(N), bt, perm, False),
        "STGCN": equivariance("STGCN", lambda a: B.STGCN(N, a), bt, perm, True),
        "DCRNN": equivariance("DCRNN", lambda a: B.DCRNN(N, a), bt, perm, True),
        "GraphWaveNet": equivariance("GraphWaveNet", lambda a: B.GraphWaveNet(N, a, emb_dim=EMB), bt, perm, True),
        "AGCRN": equivariance("AGCRN", lambda a: B.AGCRN(N, emb_dim=EMB), bt, perm, False),
        "STAEformer": equivariance("STAEformer", lambda a: STAEformer(N), bt, perm, False),
        "AMSGCN": equivariance("AMSGCN", lambda a: AMSGCN(N, emb_dim=EMB), bt, perm, False),
    }

    print("\nDo the four AMSGCN components actually do what they are named after?")

    m = AMSGCN(N, emb_dim=EMB).eval()
    with torch.no_grad():
        base = m(bt)
        alt = dict(bt); alt["adj"] = torch.eye(N).expand(BS, N, N).contiguous()
        d_adj = float((base - m(alt)).abs().max())
        alt = dict(bt); alt["tod"] = (bt["tod"] + 7) % C.SLOTS_PER_DAY
        d_tod = float((base - m(alt)).abs().max())
        alt = dict(bt); alt["dow"] = (bt["dow"] + 1) % 7
        d_dow = float((base - m(alt)).abs().max())
        m(bt)
        w = m.last_scale_weights
    print(f"  dynamic flow graph  output responds to the OD adjacency:   max|diff| = {d_adj:.3e}")
    print(f"  periodic embedding  output responds to time-of-day:        max|diff| = {d_tod:.3e}")
    print(f"  periodic embedding  output responds to day-of-week:        max|diff| = {d_dow:.3e}")
    print(f"  multi-scale         attention over {w.shape[-1]} temporal scales, "
          f"weights sum to {float(w.sum(-1).mean()):.3f} per step")

    # causality: a future input must not influence the current output
    m2 = AMSGCN(N, emb_dim=EMB).eval()
    bt_a = make_batch()
    bt_b = dict(bt_a)
    x = bt_a["x"].clone()
    x[:, -1] += 100.0                     # perturb only the most recent step
    bt_b["x"] = x
    with torch.no_grad():
        changed = float((m2(bt_a) - m2(bt_b)).abs().max())
    print(f"  temporal encoder    last input step influences the forecast: max|diff| = {changed:.3e}")

    print("\nAblation integrity:")
    nh = AMSGCN(N, emb_dim=EMB, use_heterogeneity=False)
    opt = torch.optim.Adam([p for p in nh.parameters() if p.requires_grad], lr=0.1)
    before = nh.node_emb.detach().clone()
    nh(bt).pow(2).mean().backward()
    opt.step()
    moved = float((nh.node_emb.detach() - before).abs().max())
    print(f"  w/o node heterogeneity: requires_grad={nh.node_emb.requires_grad}, "
          f"max|Δnode_emb| after one step = {moved:.6f} "
          f"-> {'genuinely frozen' if moved == 0 else 'STILL TRAINING (invalid)'}")

    ng = AMSGCN(N, emb_dim=EMB, use_dynamic_graph=False)
    ng(bt).pow(2).mean().backward()
    g = ng.node_emb.grad
    live = g is not None and float(g.abs().sum()) > 0
    print(f"  w/o dynamic flow graph: node_emb still trained = {live} "
          f"-> {'heterogeneity not confounded with the graph' if live else 'CONFOUNDED'}")

    nm = AMSGCN(N, emb_dim=EMB, use_multiscale=False).eval()
    with torch.no_grad():
        nm(bt)
    print(f"  w/o multi-scale: temporal scales reduced to {nm.last_scale_weights.shape[-1]}")

    npd = AMSGCN(N, emb_dim=EMB, use_periodic=False).eval()
    with torch.no_grad():
        b1 = npd(bt)
        alt = dict(bt); alt["tod"] = (bt["tod"] + 7) % C.SLOTS_PER_DAY
        d = float((b1 - npd(alt)).abs().max())
    print(f"  w/o periodic embedding: response to time-of-day = {d:.3e} "
          f"-> {'time input genuinely removed' if d == 0 else 'STILL PRESENT'}")

    bad = [k for k, v in results.items() if not v]
    print("\nRESULT:", "all node-axis tests passed" if not bad else f"FAILED: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
