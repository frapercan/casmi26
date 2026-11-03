"""End-to-end predictor: library search + spectrum->PLM model over DB candidates.

For one query molecule (all its spectra) it builds a candidate table:
  * library molecules within ppm (with spectral-match score L)
  * external DB molecules within ppm (no spectra; L = 0)
and scores every candidate with the trained model (z-scored per molecule).
`fuse` turns the table into the final ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from .data import ce_value
from .model import ADDUCT_IDX, SpecMolModel, sae_pred
from .sae import TopKSAE
from .search import Library, search_molecule


class CandidateDB:
    def __init__(self, path: str):
        t = pq.read_table(path, columns=["inchikey14", "smiles", "exact_mass", "pcl_entries", "in_coconut"])
        self.ik = t["inchikey14"].to_numpy(zero_copy_only=False)
        self.smiles = t["smiles"].to_numpy(zero_copy_only=False)
        self.mass = t["exact_mass"].to_numpy()
        self.pcl = t["pcl_entries"].to_numpy()
        self.coco = t["in_coconut"].to_numpy()

    def window(self, mass: float, ppm: float, max_n: int = 5000) -> np.ndarray:
        tol = max(mass * ppm * 1e-6, 0.003)
        lo = np.searchsorted(self.mass, mass - tol)
        hi = np.searchsorted(self.mass, mass + tol, side="right")
        idx = np.arange(lo, hi)
        if idx.size > max_n:  # keep the closest in mass
            idx = idx[np.argsort(np.abs(self.mass[idx] - mass))[:max_n]]
        return idx


class MolFeaturizer:
    """PLM embedding + Morgan FP for arbitrary SMILES, cached by InChIKey14."""

    def __init__(self, embedder, device):
        from .metric import ik14  # noqa: F401  (rdkit import side effects)
        from rdkit.Chem import rdFingerprintGenerator
        self.emb, self.dev = embedder, device
        self.gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __call__(self, iks, smiles):
        from rdkit import Chem
        todo = [(k, s) for k, s in zip(iks, smiles) if k not in self.cache]
        if todo:
            e = self.emb.embed([s for _, s in todo])
            for (k, s), v in zip(todo, e):
                m = Chem.MolFromSmiles(s)
                fp = self.gen.GetFingerprintAsNumPy(m).astype(np.float32) if m is not None else np.zeros(2048, np.float32)
                self.cache[k] = (v.astype(np.float32), fp)
        plm = np.stack([self.cache[k][0] for k in iks])
        fp = np.stack([self.cache[k][1] for k in iks])
        return (torch.as_tensor(plm, device=self.dev), torch.as_tensor(fp, device=self.dev))


def load_model(ckpt: str, sae_path: str = "", device="cpu"):
    ck = torch.load(ckpt, map_location=device)
    args = ck["args"]
    sae = None
    if sae_path:
        s = torch.load(sae_path, map_location=device)
        sae = TopKSAE(s["d_in"], s["n_latent"], s["k"]).to(device).eval()
        sae.load_state_dict(s["state"])
    model = SpecMolModel(d=args["d"], layers=args["layers"],
                         sae_latent=sae.enc.out_features if sae else 0).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, sae


def query_batch(q: pd.DataFrame, device, max_peaks=64) -> dict:
    B = len(q)
    mz = np.zeros((B, max_peaks), np.float32); it = np.zeros((B, max_peaks), np.float32)
    pad = np.ones((B, max_peaks), bool)
    for b, (m, i) in enumerate(zip(q.mz, q.it)):
        n = min(len(m), max_peaks)
        mz[b, :n], it[b, :n], pad[b, :n] = m[:n], i[:n], False
    t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
    ce = q["ce_mean"] if "ce_mean" in q else pd.Series([np.nan] * B)
    return {"mz": t(mz), "it": t(it), "pad": t(pad, torch.bool), "prec": t(q.precursor_mz.to_numpy()),
            "adduct": t(q.adduct.map(ADDUCT_IDX).fillna(0).to_numpy(), torch.long),
            "mode": t((q.ionization_mode == "positive").to_numpy(), torch.long),
            "ce": t(ce.map(ce_value).to_numpy())}


@torch.no_grad()
def model_scores(model, sae, b, plm, fp):
    h = model.spec(b)
    s_nce = (model.spec_emb(h) @ model.mol_emb(plm, fp).T).mean(0)
    s_jepa = (F.normalize(model.pred(h), dim=-1) @ plm.T).mean(0)
    p = torch.sigmoid(model.fp(h)).mean(0).clamp(1e-4, 1 - 1e-4)
    s_fp = fp @ p.log() + (1 - fp) @ (1 - p).log()
    out = {"nce": s_nce, "jepa": s_jepa, "fp": s_fp}
    if sae is not None:
        codes = F.normalize(sae.encode(plm), dim=-1)
        out["sae"] = (F.normalize(sae_pred(model, h), dim=-1) @ codes.T).mean(0)
    z = lambda s: ((s - s.mean()) / (s.std() + 1e-6)) if s.numel() > 1 else torch.zeros_like(s)
    return {k: z(v).float().cpu().numpy() for k, v in out.items()}


def candidate_table(q, lib: Library, db: CandidateDB, feat: MolFeaturizer, model, sae, device,
                    ppm=10.0, extra: tuple[str, str] | None = None, exclude_ik: set | None = None) -> pd.DataFrame:
    """All candidates for one query molecule with library + model features.
    extra: (ik14, smiles) injected into the DB pool (simulates a class-2 answer).
    exclude_ik: molecules to drop from the pool (e.g. simulate class 3)."""
    lib_hits = search_molecule(lib, q, ppm=ppm, k=10_000)
    rows = [(ik, smi, L, True, 0.0) for ik, smi, L in lib_hits]
    seen = {r[0] for r in rows}
    mass = float(np.median(q.mass))
    for i in db.window(mass, ppm):
        if db.ik[i] not in seen:
            rows.append((db.ik[i], db.smiles[i], 0.0, False, float(db.pcl[i]))); seen.add(db.ik[i])
    if extra is not None and extra[0] not in seen:
        rows.append((extra[0], extra[1], 0.0, False, 0.0))
    t = pd.DataFrame(rows, columns=["ik", "smiles", "L", "in_lib", "pcl"])
    if exclude_ik:
        t = t[~t.ik.isin(exclude_ik)].reset_index(drop=True)
    if len(t) == 0:
        return t
    if model is None:  # library / popularity only (no GPU needed)
        return t
    plm, fp = feat(t.ik.tolist(), t.smiles.tolist())
    for k, v in model_scores(model, sae, query_batch(q, device), plm, fp).items():
        t[k] = v
    return t


DEFAULT_W = {"nce": 1.0, "jepa": 1.0, "fp": 0.5, "sae": 1.0, "L": 4.0, "lib": 0.0, "pop": 0.1,
             "tau": 0.7, "boost": 100.0}


def fuse(t: pd.DataFrame, w: dict = DEFAULT_W) -> pd.Series:
    s = sum(w.get(k, 0) * t[k] for k in ("nce", "jepa", "fp", "sae") if k in t)
    s = s + w["L"] * t.L + w["lib"] * t.in_lib + w["pop"] * np.log1p(t.pcl)
    if len(t) and t.L.max() >= w["tau"]:  # confident library match -> library ranking first
        s = s + w["boost"] * t.L * t.in_lib
    return s
