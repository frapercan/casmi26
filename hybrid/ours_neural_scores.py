"""Drop-in for the public pipeline's neural_scores.py: our spectrum-structure model scores the exported candidates.

Same command line and output columns as the pipeline's script, so a notebook can call it where the withdrawn second
scorer was called:
    python ours_neural_scores.py BUNDLE CHECKPOINT EVIDENCE QUERIES OUTPUT [DEVICE]
BUNDLE (hybrid/build_ours_bundle.py) holds src/casmi, the checkpoint, the PLM snapshot, plm_mean.npy and
ours_config.json. Downstream layers read only neural_score and neural_ok; neural_score = (z(nce) + z(fwd)) / 2
within each molecule's list, the combination that ranks best alone on the natural-product cells.

Candidates are canonicalised (RDKit, stereo-free) before embedding: the model was trained on that format only.
Positive-mode m/z of the hidden test carry the electron-mass correction that the training libraries lack;
ours_config.json "electron_shift_pos" (+0.000548 Da) puts the queries back on the training convention.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def canon(s):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(s)
    return (Chem.MolToSmiles(m, isomericSmiles=False), m) if m is not None else (s, None)


def score(bundle, checkpoint, evidence, queries, output, device="cuda"):
    t0 = time.time()
    B = Path(bundle)
    cfg = json.loads((B / "ours_config.json").read_text())
    sys.path.insert(0, str(B / "src"))
    from rdkit.Chem import rdFingerprintGenerator
    from casmi.plm import PLMEmbedder
    from casmi.predict import load_model, model_scores, query_batch
    from casmi.search import prep_query
    dev = device if device != "cuda" or torch.cuda.is_available() else "cpu"
    model, _ = load_model(str(B / checkpoint), "", dev)
    emb = PLMEmbedder(str(B / "plm"), device=dev, mean=np.load(B / "plm_mean.npy"))
    E = pd.read_parquet(evidence)
    Q = pd.read_parquet(queries).copy()
    shift = float(cfg.get("electron_shift_pos", 0.0))
    if shift:
        pos = (Q.ionization_mode == "positive").to_numpy()
        Q.loc[pos, "precursor_mz"] = Q.loc[pos, "precursor_mz"].astype(float) + shift
        Q.loc[pos, "ms2_mzs"] = Q.loc[pos, "ms2_mzs"].map(lambda m: (np.asarray(m, float) + shift).tolist())
    q = prep_query(Q)
    q["ce_mean"] = Q.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan).to_numpy()
    QG = dict(tuple(q.groupby("molecule_id", sort=False)))

    uniq = list(dict.fromkeys(E.smiles.astype(str)))
    cm = [canon(s) for s in uniq]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = np.stack([gen.GetFingerprintAsNumPy(m).astype(np.float32) if m is not None else np.zeros(2048, np.float32) for _, m in cm])
    plm = emb.embed([c for c, _ in cm], batch_size=256).astype(np.float32)
    row = {s: i for i, s in enumerate(uniq)}
    print(f"ours: {len(uniq)} unique candidates featurised on {dev} [{time.time() - t0:.0f}s]", flush=True)

    out = []
    for mid, g in E.groupby("molecule_id", sort=False):
        g = g.copy()
        for c in ("neural_cos", "neural_fp", "neural_score", "neural_std"):
            g[c] = 0.0
        g["n_spectra"], g["n_adducts"], g["neural_ok"] = 0, 0, False
        if mid in QG and len(g):
            idx = np.array([row[s] for s in g.smiles.astype(str)])
            with torch.no_grad():
                sc = model_scores(model, None, query_batch(QG[mid], dev), torch.as_tensor(plm[idx], device=dev),
                                  torch.as_tensor(fps[idx], device=dev))
            g["neural_cos"], g["neural_fp"] = sc["nce"], sc["fp"]
            g["neural_score"] = (sc["nce"] + sc["fwd"]) / 2 if "fwd" in sc else sc["nce"]
            g["n_spectra"], g["n_adducts"], g["neural_ok"] = len(QG[mid]), QG[mid].adduct.nunique(), True
        out.append(g)
    R = pd.concat(out, ignore_index=True) if out else E.assign(neural_ok=False)
    R.to_parquet(output, index=False)
    print(f"ours: {int(R.neural_ok.sum())}/{len(R)} candidates scored, {R.molecule_id.nunique()} molecules "
          f"[{time.time() - t0:.0f}s] -> {output}", flush=True)


if __name__ == "__main__":
    score(*sys.argv[1:])
