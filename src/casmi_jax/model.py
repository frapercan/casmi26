"""JAX/Flax port of casmi.model (spectrum encoder, retrieval heads, forward world model, losses).

Same architecture and losses as the PyTorch version, written for XLA:
  * fixed shapes everywhere (batches are never de-duplicated: repeated molecules are simply extra
    positives in the multi-positive InfoNCE, so the batch size stays constant and nothing recompiles);
  * bf16 compute with fp32 parameters (`dtype`), the TPU's native mode.
"""
from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp

from casmi.spectra import ADDUCTS

N_ADDUCT = len(ADDUCTS) + 1  # 0 = unknown, same indexing as casmi.model.ADDUCT_IDX
gelu = lambda x: nn.gelu(x, approximate=False)  # torch nn.GELU default


def fourier(x, n=64, lo=1e-3, hi=1e4):
    w = jnp.exp(jnp.linspace(math.log(2 * math.pi / hi), math.log(2 * math.pi / lo), n))
    a = x[..., None] * w
    return jnp.concatenate([jnp.sin(a), jnp.cos(a)], -1)


def l2n(x, eps=1e-12):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


class MLP(nn.Module):
    hidden: int
    out: int
    drop: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, train=False):
        x = gelu(nn.Dense(self.hidden, dtype=self.dtype)(x))
        x = nn.Dropout(self.drop, deterministic=not train)(x)
        return nn.Dense(self.out, dtype=self.dtype)(x)


class Block(nn.Module):  # pre-norm transformer layer (torch norm_first=True)
    d: int
    heads: int
    drop: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, mask, train=False):
        h = nn.LayerNorm(dtype=self.dtype)(x)
        h = nn.MultiHeadDotProductAttention(self.heads, dtype=self.dtype, dropout_rate=self.drop,
                                            deterministic=not train)(h, h, mask=mask)
        x = x + nn.Dropout(self.drop, deterministic=not train)(h)
        h = nn.LayerNorm(dtype=self.dtype)(x)
        h = nn.Dense(4 * self.d, dtype=self.dtype)(h)
        h = nn.Dropout(self.drop, deterministic=not train)(gelu(h))
        h = nn.Dense(self.d, dtype=self.dtype)(h)
        return x + nn.Dropout(self.drop, deterministic=not train)(h)


class SpectrumEncoder(nn.Module):
    d: int = 256
    layers: int = 4
    heads: int = 8
    nfour: int = 64
    drop: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, b, train=False):
        f = lambda x: fourier(x, self.nfour)
        nl = jnp.maximum(b["prec"][:, None] - b["mz"], 0)
        tok = jnp.concatenate([f(b["mz"]), f(nl), b["it"][..., None]], -1)
        tok = nn.Dense(self.d, dtype=self.dtype)(gelu(nn.Dense(self.d, dtype=self.dtype)(tok)))
        p = jnp.concatenate([f(b["prec"]), (b["mode"].astype(jnp.float32) * 2 - 1)[:, None], (b["ce"] / 100)[:, None]], -1)
        p = nn.Dense(self.d, dtype=self.dtype)(gelu(nn.Dense(self.d, dtype=self.dtype)(p)))
        p = p + nn.Embed(N_ADDUCT, self.d, dtype=self.dtype)(b["adduct"])
        x = jnp.concatenate([p[:, None], tok], 1)
        keep = jnp.concatenate([jnp.ones_like(b["pad"][:, :1]), ~b["pad"]], 1)      # True = attend
        mask = nn.make_attention_mask(keep, keep)
        for _ in range(self.layers):
            x = Block(self.d, self.heads, self.drop, self.dtype)(x, mask, train)
        return nn.LayerNorm(dtype=self.dtype)(x)[:, 0]


class SpecMolModel(nn.Module):
    plm_dim: int = 384
    fp_bits: int = 2048
    d: int = 256
    emb: int = 256
    layers: int = 4
    heads: int = 8
    drop: float = 0.1
    fwd: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        dt = self.dtype
        self.enc = SpectrumEncoder(self.d, self.layers, self.heads, drop=self.drop, dtype=dt)
        self.proj_s = MLP(2 * self.d, self.emb, self.drop, dt)
        self.pred = MLP(2 * self.d, self.plm_dim, self.drop, dt)
        self.fp = MLP(2 * self.d, self.fp_bits, self.drop, dt)
        self.proj_m = MLP(2 * self.d, self.emb, self.drop, dt)
        self.logit_scale = self.param("logit_scale", lambda _: jnp.asarray(math.log(1 / 0.07), jnp.float32))
        if self.fwd:
            self.cond_add = nn.Embed(N_ADDUCT, self.d, dtype=dt)
            self.cond_num = MLP(self.d, self.d, self.drop, dt)
            self.fwd_m = MLP(2 * self.d, self.emb, self.drop, dt)
            self.film = nn.Dense(2 * self.emb, dtype=dt)

    def spec(self, b, train=False):
        return self.enc(b, train)

    def spec_emb(self, h, train=False):
        return l2n(self.proj_s(h, train).astype(jnp.float32))

    def mol_emb(self, plm, fp, train=False):
        return l2n(self.proj_m(jnp.concatenate([plm, fp], -1), train).astype(jnp.float32))

    def fp_logits(self, h, train=False):
        return self.fp(h, train).astype(jnp.float32)

    def plm_pred(self, h, train=False):
        return self.pred(h, train).astype(jnp.float32)

    def fwd_emb(self, plm, fp, b, train=False):
        """Predicted spectrum latent of every molecule under every spectrum's conditions -> (B,M,emb)."""
        num = jnp.stack([b["mode"].astype(jnp.float32) * 2 - 1, b["ce"] / 100], -1)
        c = self.cond_add(b["adduct"]) + self.cond_num(num, train)
        g = self.fwd_m(jnp.concatenate([plm, fp], -1), train)
        if g.ndim == 2:
            g = g[None]
        gam, bet = jnp.split(self.film(c), 2, -1)
        return l2n((g * (1 + gam[:, None]) + bet[:, None]).astype(jnp.float32))

    def __call__(self, b, train=False):  # used only for init
        h = self.spec(b, train)
        out = (self.spec_emb(h), self.mol_emb(b["plm"], b["fp"]), self.fp_logits(h), self.plm_pred(h))
        if self.fwd:
            out = out + (self.fwd_emb(b["plm"], b["fp"], b),)
        return out


def _ms_nce(logits, pos):
    """Multi-positive InfoNCE: -log sum_pos softmax."""
    lse_pos = jax.nn.logsumexp(jnp.where(pos, logits, -1e4), 1)
    return -(lse_pos - jax.nn.logsumexp(logits, 1)).mean()


def losses(model: SpecMolModel, params, b, rng, w_nce=1.0, w_pred=1.0, w_fp=0.5, w_hard=1.0, w_fwd=1.0, train=True):
    """Same objective as casmi.model.losses. b holds B spectra; rows sharing mol_id are positives."""
    m = model.bind({"params": params}, rngs={"dropout": rng})
    h = m.spec(b, train)
    zs = m.spec_emb(h, train)
    zm = m.mol_emb(b["plm"], b["fp"], train)
    scale = jnp.minimum(jnp.exp(params["logit_scale"]), 100.0)
    logits = scale * zs @ zm.T
    same = b["mol_id"][None, :] == b["mol_id"][:, None]
    nce = (_ms_nce(logits, same) + _ms_nce(logits.T, same.T)) / 2
    pp = m.plm_pred(h, train)
    pred = 1 - (l2n(pp) * l2n(b["plm"])).sum(-1).mean()
    fp = optax_bce(m.fp_logits(h, train), b["fp"])
    tot = w_nce * nce + w_pred * pred + w_fp * fp
    info = {"nce": nce, "pred": pred, "fp": fp,
            "acc": (logits.argmax(1) == jnp.arange(len(zs))).mean()}
    if "neg_plm" in b:
        ok = b["neg_ok"]
        zn = m.mol_emb(b["neg_plm"], b["neg_fp"], train)
        s_pos = (zs * zm).sum(-1, keepdims=True)
        s_neg = jnp.where(ok, jnp.einsum("be,bke->bk", zs, zn), -1e4)
        hl = scale * jnp.concatenate([s_pos, s_neg], 1)
        hard = -(hl[:, 0] - jax.nn.logsumexp(hl, 1)).mean()
        tot = tot + w_hard * hard
        info["hard"] = hard
    if model.fwd:
        fl = jnp.einsum("be,bme->bm", zs, m.fwd_emb(b["plm"], b["fp"], b, train))
        pos = same
        if "neg_plm" in b:
            fn = jnp.einsum("be,bke->bk", zs, m.fwd_emb(b["neg_plm"], b["neg_fp"], b, train))
            fl = jnp.concatenate([fl, jnp.where(b["neg_ok"], fn, -1e4)], 1)
            pos = jnp.concatenate([same, jnp.zeros_like(b["neg_ok"])], 1)
        fwd = _ms_nce(scale * fl, pos)
        tot = tot + w_fwd * fwd
        info["fwd"] = fwd
    return tot, info


def optax_bce(logits, y):
    return (jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits)))).mean()
