"""Why does the class-3 generator miss? For molecules whose truth was not generated, look at the analogs:
how close (Tanimoto) the nearest analog in the top-K spectral neighbours is, and the formula difference
between that analog and the truth (which edit would have been needed).

  python scripts/diag_class3.py --cell /home/user/work/val3 --stats /home/user/work/val3/gen_a30 --n 80
"""
import argparse, glob, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from gen_class3 import binned, library_matrix, Library, prep_query, Chem, BulkTanimotoSimilarity, rdFingerprintGenerator  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402
from collections import Counter


def formula_counts(m):
    m = Chem.AddHs(m); c = Counter(a.GetSymbol() for a in m.GetAtoms()); return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--lib", default="/home/user/work/lib"); ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); t0 = time.time()
    St = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{a.stats}/stats_*.parquet")])
    A = pd.read_parquet(f"{a.cell}/answers.parquet").set_index("molecule_id")
    Q = pd.read_parquet(f"{a.cell}/queries.parquet")
    hide = set(open(f"{a.cell}/exclude_all_ik.txt").read().split())
    lib = Library.load(a.lib); lib = lib.subset(~lib.spectra.inchikey14.isin(hide).to_numpy())
    F, N = library_matrix(lib); spec_ik = lib.spectra.inchikey14.to_numpy(); mode_ = lib.spectra.ionization_mode.to_numpy()
    smi_of = dict(zip(lib.molecules.inchikey14, lib.molecules.smiles))
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    q = prep_query(Q)
    rng = np.random.default_rng(0)
    miss = St.molecule_id[~St.generated].tolist(); hit = St.molecule_id[St.generated].tolist()
    pick = list(rng.choice(miss, min(a.n, len(miss)), replace=False)) + list(rng.choice(hit, min(a.n // 4, len(hit)), replace=False))
    rows = []
    for k, mid in enumerate(pick):
        g = q[q.molecule_id == mid]; tm = Chem.MolFromSmiles(A.smiles[mid])
        sim = np.zeros(len(spec_ik), np.float32)
        for mz, it, prec, mo in zip(g.mz, g.it, g.precursor_mz, g.ionization_mode):
            f, n = binned(mz, it, prec); s = F @ f + 0.5 * (N @ n); s[mode_ != mo] = 0; sim = np.maximum(sim, s)
        best = pd.Series(sim).groupby(spec_ik).max().sort_values(ascending=False).head(300)
        par = [Chem.MolFromSmiles(smi_of[i]) for i in best.index]
        tan = np.array(BulkTanimotoSimilarity(fpg.GetFingerprint(tm), [fpg.GetFingerprint(m) if m else fpg.GetFingerprint(Chem.MolFromSmiles("C")) for m in par]))
        j = int(np.argmax(tan[:30])); jj = int(np.argmax(tan))
        ft, fa = formula_counts(tm), formula_counts(par[j])
        diff = {e: ft.get(e, 0) - fa.get(e, 0) for e in set(ft) | set(fa) if ft.get(e, 0) != fa.get(e, 0)}
        rows.append(dict(molecule_id=mid, generated=mid in hit, max_tan30=tan[:30].max(), max_tan100=tan[:100].max(), max_tan300=tan.max(),
                         rank_best300=jj + 1, n_tan07_30=int((tan[:30] >= 0.7).sum()), heavy=tm.GetNumHeavyAtoms(),
                         diff=" ".join(f"{e}{v:+d}" for e, v in sorted(diff.items())) or "isomer",
                         truth_formula=CalcMolFormula(tm), np_like=None))
        if (k + 1) % 20 == 0:
            print(f"{k+1}/{len(pick)} [{time.time()-t0:.0f}s]", flush=True)
    R = pd.DataFrame(rows); R.to_parquet(a.out, index=False)
    for gen, d in R.groupby("generated"):
        print(f"\n== generated={gen} ({len(d)}): max Tanimoto of analogs top30 / top100 / top300 (median): "
              f"{d.max_tan30.median():.2f} / {d.max_tan100.median():.2f} / {d.max_tan300.median():.2f}")
        for th in [0.5, 0.7, 0.85]:
            print(f"   frac with an analog >= {th}: top30 {(d.max_tan30 >= th).mean():.2f}  top100 {(d.max_tan100 >= th).mean():.2f}  top300 {(d.max_tan300 >= th).mean():.2f}")
        print("   formula diff truth - nearest top-30 analog:", d["diff"].value_counts().head(15).to_dict())


if __name__ == "__main__":
    main()
