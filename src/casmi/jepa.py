"""Self-supervised JEPA pretraining for the spectrum encoder.

Context encoder sees the spectrum with some peaks replaced by a mask token
(contiguous m/z blocks); an EMA target encoder sees the full spectrum. A
narrow predictor maps context tokens to the target's representations of the
masked peaks and of the precursor (CLS) token. Loss is on latents, not peaks.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SpectrumEncoder


def block_mask(mz: torch.Tensor, pad: torch.Tensor, n_blocks: int = 3, frac: float = 0.4,
               gen: torch.Generator | None = None) -> torch.Tensor:
    """Mask peaks falling in random m/z windows covering ~frac of the m/z span."""
    B, N = mz.shape
    big = torch.where(pad, torch.full_like(mz, -1.0), mz)
    hi = big.max(1).values.clamp(min=1.0)
    lo = torch.where(pad, torch.full_like(mz, 1e9), mz).min(1).values.clamp(max=hi - 1e-3)
    width = (hi - lo) * frac / n_blocks
    mask = torch.zeros_like(pad)
    for _ in range(n_blocks):
        c = lo + torch.rand(B, device=mz.device, generator=gen) * (hi - lo)
        mask |= (mz - c.unsqueeze(1)).abs() <= (width / 2).unsqueeze(1)
    mask &= ~pad
    # guarantee at least one masked and one visible peak where possible
    n_valid = (~pad).sum(1)
    none = (mask.sum(1) == 0) & (n_valid > 1)
    if none.any():
        first = (~pad).float().argmax(1)
        mask[none, first[none]] = True
    allm = mask.sum(1) == n_valid
    if allm.any():
        first = mask.float().argmax(1)
        mask[allm, first[allm]] = False
    return mask


class JEPA(nn.Module):
    """query="mz" (v2): masked peaks are hidden from the context encoder (key padding) and the predictor gets one
    query per masked peak = learned token + embedding of that peak's m/z and neutral loss (no intensity), so it
    knows WHICH peak to predict. query="none" (v1, kept to reproduce old runs): masked peaks become one shared
    position-free token, so every masked prediction in a spectrum is identical and the token loss only learns a
    mean. ctx_free=True is the shortcut control: the predictor sees the queries but no context at all."""

    def __init__(self, d: int = 256, layers: int = 4, pred_layers: int = 2, ema: float = 0.996,
                 query: str = "mz", ctx_free: bool = False, **kw):
        super().__init__()
        assert query in ("mz", "none") and not (ctx_free and query == "none")
        self.ctx = SpectrumEncoder(d=d, layers=layers, **kw)
        self.tgt = copy.deepcopy(self.ctx).requires_grad_(False)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, 0.0, activation="gelu", batch_first=True, norm_first=True)
        self.pred = nn.TransformerEncoder(layer, pred_layers)
        self.pred_out = nn.Linear(d, d)
        self.ema = ema
        self.query, self.ctx_free = query, ctx_free
        if query == "mz":
            f = 2 * self.ctx.four.w.numel()
            self.q_tok = nn.Parameter(torch.zeros(d))
            self.qpos = nn.Sequential(nn.Linear(2 * f, d), nn.GELU(), nn.Linear(d, d))

    @torch.no_grad()
    def update_target(self, m: float | None = None):
        m = self.ema if m is None else m
        for pt, pc in zip(self.tgt.parameters(), self.ctx.parameters()):
            pt.mul_(m).add_(pc.detach(), alpha=1 - m)

    def loss(self, b: dict):
        args = (b["mz"], b["it"], b["pad"], b["prec"], b["adduct"], b["mode"], b["ce"])
        mask = block_mask(b["mz"], b["pad"])
        self.tgt.eval()   # deterministic targets (no dropout in the EMA encoder)
        with torch.no_grad():
            tc, tt = self.tgt(*args, return_tokens=True)
            tc, tt = F.layer_norm(tc, tc.shape[-1:]), F.layer_norm(tt, tt.shape[-1:])
        kpm = torch.cat([torch.zeros_like(b["pad"][:, :1]), b["pad"]], 1)
        if self.query == "none":
            cc, ct = self.ctx(*args, peak_mask=mask, return_tokens=True)
            x = torch.cat([cc.unsqueeze(1), ct], 1)
        else:
            four = self.ctx.four
            nl = (b["prec"].unsqueeze(1) - b["mz"]).clamp(min=0)
            q = self.q_tok + self.qpos(torch.cat([four(b["mz"]), four(nl)], -1))
            if self.ctx_free:
                cc = torch.zeros_like(tc)
                x = torch.cat([cc.unsqueeze(1), torch.where(mask.unsqueeze(-1), q, torch.zeros_like(q))], 1)
                kpm = torch.cat([torch.zeros_like(b["pad"][:, :1]), b["pad"] | ~mask], 1)
            else:
                cc, ct = self.ctx(b["mz"], b["it"], b["pad"] | mask, *args[3:], return_tokens=True)
                x = torch.cat([cc.unsqueeze(1), torch.where(mask.unsqueeze(-1), q, ct)], 1)
        p = self.pred_out(self.pred(x, src_key_padding_mask=kpm))
        pm, tm = p[:, 1:][mask], tt[mask]
        tok_loss = F.smooth_l1_loss(pm, tm)
        cls_loss = F.smooth_l1_loss(p[:, 0], tc) if not self.ctx_free else torch.zeros((), device=tc.device)
        std = cc.float().std(0)
        var_loss = F.relu(1 - std).mean() if not self.ctx_free else torch.zeros((), device=tc.device)
        loss = tok_loss + cls_loss + 0.1 * var_loss
        with torch.no_grad():   # within-spectrum spread of masked predictions vs targets (v1 bug -> 0)
            spread = lambda z: torch.stack([z[i][mask[i]].float().std(0).mean() for i in range(len(z)) if mask[i].sum() > 1]).mean()
            ratio = spread(p[:, 1:]) / spread(tt).clamp(min=1e-6)
        return loss, {"tok": tok_loss.detach(), "cls": cls_loss.detach(), "std": std.mean().detach(),
                      "masked": mask.float().sum() / (~b["pad"]).sum(), "spread": ratio}
