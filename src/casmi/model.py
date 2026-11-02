"""Spectrum encoder + molecule head trained to meet in the PLM latent space.

Spectrum side: Transformer over peak tokens (Fourier m/z + neutral loss +
intensity) with a precursor token carrying mass, adduct, mode and collision
energy. Outputs a CLS vector h.

Heads on h:
  * proj_s  -> shared retrieval space (InfoNCE against molecules)
  * pred    -> predicted PLM embedding (JEPA-style latent prediction)
  * fp      -> Morgan fingerprint logits (sparse multi-label target)
Molecule side: proj_m(PLM embedding, fingerprint) -> shared retrieval space.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectra import ADDUCTS

ADDUCT_IDX = {a: i + 1 for i, a in enumerate(sorted(ADDUCTS))}  # 0 = unknown


class Fourier(nn.Module):
    """Multi-scale sinusoidal features of a mass (Da), resolving ~1e-3 Da to 1e3 Da."""

    def __init__(self, n: int = 64, lo: float = 1e-3, hi: float = 1e4):
        super().__init__()
        w = torch.exp(torch.linspace(math.log(2 * math.pi / hi), math.log(2 * math.pi / lo), n))
        self.register_buffer("w", w)

    def forward(self, x):  # (...,) -> (..., 2n)
        a = x.unsqueeze(-1) * self.w
        return torch.cat([a.sin(), a.cos()], -1)


class SpectrumEncoder(nn.Module):
    def __init__(self, d: int = 256, layers: int = 4, heads: int = 8, nfour: int = 64, drop: float = 0.1):
        super().__init__()
        self.four = Fourier(nfour)
        f = 2 * nfour
        self.peak = nn.Sequential(nn.Linear(2 * f + 1, d), nn.GELU(), nn.Linear(d, d))
        self.prec = nn.Sequential(nn.Linear(f + 2, d), nn.GELU(), nn.Linear(d, d))
        self.adduct = nn.Embedding(len(ADDUCT_IDX) + 1, d)
        self.mask_tok = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.d = d

    def forward(self, mz, it, pad, prec_mz, adduct, mode, ce, peak_mask=None, return_tokens=False):
        """mz,it,pad: (B,N); prec_mz,ce: (B,); adduct,mode: (B,) long.
        peak_mask: (B,N) bool, True = replace token by mask token (JEPA context)."""
        nl = (prec_mz.unsqueeze(1) - mz).clamp(min=0)
        tok = self.peak(torch.cat([self.four(mz), self.four(nl), it.unsqueeze(-1)], -1))
        if peak_mask is not None:
            tok = torch.where(peak_mask.unsqueeze(-1), self.mask_tok.expand_as(tok), tok)
        p = self.prec(torch.cat([self.four(prec_mz), (mode.float() * 2 - 1).unsqueeze(-1),
                                 (ce / 100).unsqueeze(-1)], -1)) + self.adduct(adduct)
        x = torch.cat([p.unsqueeze(1), tok], 1)
        kpm = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        x = self.norm(self.tf(x, src_key_padding_mask=kpm))
        return (x[:, 0], x[:, 1:]) if return_tokens else x[:, 0]


def mlp(i, h, o, drop=0.1):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, o))


class SpecMolModel(nn.Module):
    def __init__(self, plm_dim: int = 384, fp_bits: int = 2048, d: int = 256, emb: int = 256, **enc_kw):
        super().__init__()
        self.enc = SpectrumEncoder(d=d, **enc_kw)
        self.proj_s = mlp(d, 2 * d, emb)
        self.pred = mlp(d, 2 * d, plm_dim)
        self.fp = mlp(d, 2 * d, fp_bits)
        self.proj_m = mlp(plm_dim + fp_bits, 2 * d, emb)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def spec(self, batch):
        h = self.enc(batch["mz"], batch["it"], batch["pad"], batch["prec"], batch["adduct"],
                     batch["mode"], batch["ce"])
        return h

    def spec_emb(self, h):
        return F.normalize(self.proj_s(h), dim=-1)

    def mol_emb(self, plm, fp):
        return F.normalize(self.proj_m(torch.cat([plm, fp], -1)), dim=-1)


def losses(model: SpecMolModel, batch, w_nce=1.0, w_pred=1.0, w_fp=0.5):
    """batch carries spectra (B) and their molecules' plm (B,P) / fp (B,F).
    Rows with the same molecule id are treated as positives of each other."""
    h = model.spec(batch)
    zs = model.spec_emb(h)
    zm = model.mol_emb(batch["plm"], batch["fp"])
    logits = model.logit_scale.exp().clamp(max=100) * zs @ zm.T
    same = batch["mol_id"].unsqueeze(0) == batch["mol_id"].unsqueeze(1)
    # multi-positive InfoNCE (spectrum -> molecule and molecule -> spectrum)
    lp = logits.masked_fill(~same, -1e4).logsumexp(1) - logits.logsumexp(1)
    lp2 = logits.T.masked_fill(~same, -1e4).logsumexp(1) - logits.T.logsumexp(1)
    nce = -(lp.mean() + lp2.mean()) / 2
    pred = 1 - F.cosine_similarity(model.pred(h), batch["plm"], dim=-1).mean()
    fp = F.binary_cross_entropy_with_logits(model.fp(h), batch["fp"])
    tot = w_nce * nce + w_pred * pred + w_fp * fp
    with torch.no_grad():
        acc = (logits.argmax(1) == torch.arange(len(zs), device=zs.device)).float().mean()
    return tot, {"nce": nce.item(), "pred": pred.item(), "fp": fp.item(), "acc": acc.item()}
