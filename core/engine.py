"""Training and evaluation loop shared by every neural model.

Adapted from the research pipeline. Course additions (written for ECE 592):
  - `loss="pinball"` with a quantile level q, so the same network can be trained
    for an upper quantile of demand instead of the median;
  - early stopping on the validation value of the SAME loss being optimized;
  - validation predictions are returned as well as test predictions, because
    the decision-aware experiment selects q on the validation day.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .metrics import per_horizon


class PinballLoss(nn.Module):
    """Quantile (pinball) loss. q = 0.5 gives half the L1 loss, so the
    ordinary L1-trained model is the q = 0.5 member of this family."""

    def __init__(self, q: float):
        super().__init__()
        self.q = float(q)

    def forward(self, pred, target):
        e = target - pred
        return torch.maximum(self.q * e, (self.q - 1.0) * e).mean()


def pinball_np(p, y, q):
    e = y - p
    return float(np.maximum(q * e, (q - 1.0) * e).mean())


def make_loader(ds, shuffle):
    return DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=shuffle, drop_last=False)


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def collect(model, loader, scaler, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        p = model(batch)
        preds.append(p.cpu().numpy())
        labels.append(batch["y"].cpu().numpy())
    p = np.concatenate(preds)[..., 0]            # [S, P, N]
    y = np.concatenate(labels)[..., 0]
    return scaler.inverse(p), scaler.inverse(y)


def train_model(model, bundle, seed, device=C.DEVICE, max_epochs=C.MAX_EPOCHS,
                patience=C.PATIENCE, loss="l1", q=0.5, verbose=False, ckpt_path=None):
    """Returns (test metrics, test preds, test labels, val preds, val labels).

    With `ckpt_path`, training state is saved after every epoch and a later call
    with the same path resumes from it (for machines that cap process lifetime).
    """
    C.seed_everything(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(3, patience // 3))
    if loss == "l1":
        crit = nn.L1Loss()
        val_score = lambda p, y: float(np.abs(p - y).mean())
    elif loss == "pinball":
        crit = PinballLoss(q)
        val_score = lambda p, y: pinball_np(p, y, q)
    else:
        raise ValueError(loss)

    tr = make_loader(bundle.train, True)
    va = make_loader(bundle.val, False)
    te = make_loader(bundle.test, False)

    best, best_state, bad, epochs_run, start, elapsed = float("inf"), None, 0, 0, 1, 0.0
    if ckpt_path is not None and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        best, best_state, bad, epochs_run, elapsed = ck["best"], ck["best_state"], ck["bad"], ck["epoch"], ck["elapsed"]
        torch.set_rng_state(ck["rng"])
        start = epochs_run + 1
        if ck.get("stopped"):
            start = max_epochs + 1                  # early stopping had already fired
    t0 = time.time() - elapsed
    stopped = False
    for epoch in range(start, max_epochs + 1):
        model.train()
        for batch in tr:
            batch = _to_device(batch, device)
            opt.zero_grad()
            l = crit(model(batch), batch["y"])
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
        epochs_run = epoch

        p, y = collect(model, va, bundle.scaler, device)
        v = val_score(p, y)
        sched.step(v)
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                stopped = True
        if ckpt_path is not None:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "best": best, "best_state": best_state, "bad": bad, "epoch": epoch,
                        "elapsed": time.time() - t0, "rng": torch.get_rng_state(), "stopped": stopped},
                       ckpt_path)
        if stopped:
            break
        if verbose and epoch % 10 == 0:
            print(f"      epoch {epoch:3d}  val {v:.4f}  best {best:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    p_val, y_val = collect(model, va, bundle.scaler, device)
    p, y = collect(model, te, bundle.scaler, device)
    metrics = per_horizon(p, y)
    metrics["val_loss"] = best
    metrics["epochs"] = epochs_run
    metrics["train_sec"] = time.time() - t0
    metrics["params"] = sum(t.numel() for t in model.parameters() if t.requires_grad)
    return metrics, p, y, p_val, y_val
