"""Score candidates with a JAX/Flax checkpoint (same heads and z-scoring as casmi.predict.model_scores)."""
from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from .model import SpecMolModel, l2n


class JaxScorer:
    def __init__(self, run_dir: str):
        a = json.load(open(os.path.join(run_dir, "args.json")))
        self.model = SpecMolModel(d=a["d"], layers=a["layers"], fwd=a.get("fwd", False),
                                  dtype=jnp.bfloat16 if a.get("bf16") else jnp.float32)
        raw = open(os.path.join(run_dir, "best.msgpack"), "rb").read()
        self.params = serialization.msgpack_restore(raw)
        self.has_fwd = a.get("fwd", False)
        self._f = jax.jit(self._scores)

    def _scores(self, params, b, plm, fp):
        m = self.model.bind({"params": params})
        h = m.spec(b)
        zs = m.spec_emb(h)
        out = {"nce": (zs @ m.mol_emb(plm, fp).T).mean(0),
               "jepa": (l2n(m.plm_pred(h)) @ l2n(plm).T).mean(0)}
        p = jnp.clip(jax.nn.sigmoid(m.fp_logits(h)).mean(0), 1e-4, 1 - 1e-4)
        out["fp"] = fp @ jnp.log(p) + (1 - fp) @ jnp.log(1 - p)
        if self.has_fwd:
            out["fwd"] = jnp.einsum("se,sme->sm", zs, m.fwd_emb(plm, fp, b)).mean(0)
        return out

    def __call__(self, b: dict, plm, fp) -> dict:
        """b: torch or numpy query batch from casmi.predict.query_batch; plm, fp: (M, .) candidates."""
        to = lambda x: jnp.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x))
        jb = {k: to(b[k]) for k in ("mz", "it", "pad", "prec", "adduct", "mode", "ce")}
        out = self._f(self.params, jb, to(plm).astype(jnp.float32), to(fp).astype(jnp.float32))
        z = lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-6) if s.size > 1 else s * 0
        return {k: np.asarray(z(np.asarray(v, np.float64))) for k, v in out.items()}
