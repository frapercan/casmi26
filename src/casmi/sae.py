"""Top-k sparse autoencoder over PLM molecule embeddings.

Gives every molecule a sparse code (k active of n_latent dictionary atoms);
the spectrum model learns to predict these codes, and candidates are scored
by agreement between predicted and actual codes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_in: int = 384, n_latent: int = 4096, k: int = 32):
        super().__init__()
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.enc = nn.Linear(d_in, n_latent)
        self.dec = nn.Linear(n_latent, d_in, bias=False)
        with torch.no_grad():
            self.dec.weight.copy_(F.normalize(torch.randn(d_in, n_latent), dim=0))
            self.enc.weight.copy_(self.dec.weight.T)
            self.enc.bias.zero_()

    def encode(self, x):
        a = F.relu(self.enc(x - self.b_pre))
        v, i = a.topk(self.k, dim=-1)
        return torch.zeros_like(a).scatter_(-1, i, v)

    def forward(self, x):
        z = self.encode(x)
        return self.dec(z) + self.b_pre, z

    @torch.no_grad()
    def normalize_decoder(self):
        self.dec.weight.data = F.normalize(self.dec.weight.data, dim=0)


def train_sae(x: torch.Tensor, n_latent=4096, k=32, steps=20000, batch=4096, lr=4e-4, device="cpu", log=print):
    """x: (N, d) float tensor (CPU ok). Returns trained SAE and dead-latent fraction."""
    sae = TopKSAE(x.shape[1], n_latent, k).to(device)
    sae.b_pre.data = x[:100_000].float().mean(0).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    fired = torch.zeros(n_latent, dtype=torch.bool, device=device)
    for s in range(steps):
        xb = x[torch.randint(len(x), (batch,))].float().to(device)
        rec, z = sae(xb)
        loss = F.mse_loss(rec, xb)
        opt.zero_grad(); loss.backward(); opt.step(); sae.normalize_decoder()
        fired |= (z > 0).any(0)
        if (s + 1) % 1000 == 0:
            fvu = loss.item() / xb.var(0).sum().item() * xb.shape[1]
            log(f"sae step {s+1} mse {loss.item():.5f} fvu {fvu:.3f} alive {fired.float().mean():.3f}")
            fired.zero_()
    return sae
