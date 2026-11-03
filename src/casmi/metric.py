"""MRR@25 replicating the competition metric.

A prediction matches when the InChIKey first block (InChIKey14) of the
tautomer-canonicalised molecule equals the answer's.
"""
from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
_TAUT = rdMolStandardize.TautomerEnumerator()
MAX_K = 25


@lru_cache(maxsize=500_000)
def ik14(smiles: str) -> str | None:
    """InChIKey14 exactly as the official scorer (casmi-mean-reciprocal-rank v13):
    stereo removed *before* tautomer canonicalisation (a stereocentre can pin a double
    bond and change the canonical tautomer); None if unparsable or on RDKit errors."""
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        key = Chem.MolToInchiKey(_TAUT.Canonicalize(mol))
        return key[:14] if key else None
    except (ValueError, RuntimeError, TypeError):
        return None


def reciprocal_rank(pred_smiles: list[str], true_ik14: str, k: int = MAX_K) -> float:
    for i, s in enumerate(pred_smiles[:k]):
        if ik14(s) == true_ik14:
            return 1.0 / (i + 1)
    return 0.0


def reciprocal_rank_ik(pred_ik14: list[str], true_ik14: str, k: int = MAX_K) -> float:
    """Fast path when candidates already carry their (canonical) InChIKey14."""
    for i, key in enumerate(pred_ik14[:k]):
        if key == true_ik14:
            return 1.0 / (i + 1)
    return 0.0


def mrr(preds: dict[str, list[str]], truth: dict[str, str], k: int = MAX_K) -> float:
    """preds: molecule_id -> ranked SMILES; truth: molecule_id -> true SMILES."""
    tot = 0.0
    for mid, true_smi in truth.items():
        tot += reciprocal_rank(preds.get(mid, []), ik14(true_smi), k)
    return tot / max(len(truth), 1)
