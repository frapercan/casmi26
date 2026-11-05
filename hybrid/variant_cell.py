# ── casmi26: ship one of the pipeline's own variants as the final submission (chosen on the honest evaluation) ──
VARIANT = __VARIANT__
import os, shutil, pandas as pd
try:
    src = f"/kaggle/working/variants/submission_{VARIANT}.csv"
    s, v = pd.read_csv("/kaggle/working/submission.csv", dtype=str), pd.read_csv(src, dtype=str)
    assert list(v.columns) == list(s.columns) and set(v.molecule_id) == set(s.molecule_id), "variant file does not match"
    shutil.copyfile("/kaggle/working/submission.csv", "/kaggle/working/submission_upstream.csv")
    v.set_index("molecule_id").loc[s.molecule_id].reset_index().to_csv("/kaggle/working/submission.csv", index=False)
    print(f"OURS: variant {VARIANT} shipped as submission.csv ({len(v)} rows)")
except Exception as e:
    print("OURS: variant not shipped, upstream submission kept:", repr(e))
