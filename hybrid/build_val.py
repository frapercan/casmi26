"""Run the public 0.480 pipeline on our hidden-test-like validation set (a normal kernel run,
not a submission), then score its lists and sweep our model's re-rank weight.

  python hybrid/build_val.py --upstream /path/base480.ipynb --slug casmi26-hybrid-val --n1 80 --n2 170 [--dry out.ipynb]

The pipeline's competition-folder discovery is patched to /kaggle/working/valcomp (built by
val_pre_cell.py); its pinned inner notebook is re-encoded with the same three path patches and
its pinned SHA-256 updated. Nothing else in the pipeline changes.
"""
import argparse, ast, base64, copy, hashlib, io, json, os, tarfile
import requests
ROOT = os.path.join(os.path.dirname(__file__), "..")
VC = "/kaggle/working/valcomp"


def sub1(s, old, new, what):
    n = s.count(old)
    assert n == 1, f"{what}: expected 1 match, got {n}"
    return s.replace(old, new)


def patch_frozen(fr: bytes) -> bytes:
    nb = json.loads(fr)
    done = set()
    P = [("data", 'Path("/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra"),', f'Path("{VC}"),'),
         ("comp", "COMP = os.path.dirname(find('test.parquet'))", f"COMP = '{VC}'"),
         ("preflight", "    assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}\n", "    pass\n")]
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        for name, old, new in P:
            if old in s:
                s = sub1(s, old, new, name); done.add(name)
        c["source"] = s
    assert done == {p[0] for p in P}, done
    return json.dumps(nb).encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--slug", default="casmi26-hybrid-val")
    ap.add_argument("--n1", type=int, default=80)
    ap.add_argument("--n2", type=int, default=170)
    ap.add_argument("--dry", default="")
    ap.add_argument("--extra-src", action="append", default=[], help="extra kernel source (owner/slug)")
    ap.add_argument("--ext", type=int, default=0, help="honest eval: N external Enveda-180 molecules absent from train")
    ap.add_argument("--val3", action="store_true", help="honest eval, class 3 (private dataset casmi26-class3-data)")
    a = ap.parse_args()
    nb = copy.deepcopy(json.load(open(a.upstream)))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    # 1) pinned inner pipeline
    cell = next(c for c in code if "FROZEN_B64 = '" in "".join(c["source"]))
    src = "".join(cell["source"])
    tree = ast.parse(src)
    vals = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("FROZEN_B64", "FROZEN_SHA256")}
    fr = base64.b64decode(vals["FROZEN_B64"])
    assert hashlib.sha256(fr).hexdigest() == vals["FROZEN_SHA256"]
    new = patch_frozen(fr)
    src = sub1(src, vals["FROZEN_B64"], base64.b64encode(new).decode(), "b64")
    src = sub1(src, vals["FROZEN_SHA256"], hashlib.sha256(new).hexdigest(), "sha")
    cell["source"] = src
    # 2) outer competition-folder discovery
    c2 = next(c for c in code if "COMP = next((p for p in (INPUT / 'competitions/" in "".join(c["source"]))
    c2["source"] = sub1("".join(c2["source"]), "COMP = next((p for p in (INPUT / 'competitions/",
                        f"COMP = next((p for p in (Path('{VC}'), INPUT / 'competitions/", "outer comp")
    # 3) our cells
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(os.path.join(ROOT, "src"), arcname="src", filter=lambda t: None if "__pycache__" in t.name else t)
    if a.val3:
        pre = open(os.path.join(ROOT, "hybrid", "val3_pre_cell.py")).read()
    elif a.ext:
        pre = open(os.path.join(ROOT, "hybrid", "ext_pre_cell.py")).read().replace("__N_MOL__", str(a.ext))
    else:
        pre = open(os.path.join(ROOT, "hybrid", "val_pre_cell.py")).read().replace("__N_CLS__", repr({1: a.n1, 2: a.n2}))
    post = open(os.path.join(ROOT, "hybrid", "val_post_cell.py")).read().replace("__PAYLOAD__", base64.b64encode(buf.getvalue()).decode())
    mk = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
    nb["cells"].insert(0, mk(pre))
    nb["cells"].append(mk(post))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    if a.dry:
        json.dump(nb, open(a.dry, "w")); print("wrote", a.dry); return
    r = requests.post("https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel", headers={"Authorization": "Bearer proxy"}, timeout=300, json={
        "slug": f"xaxipirulazo/{a.slug}", "newTitle": a.slug, "text": json.dumps(nb), "language": "python",
        "kernelType": "notebook", "isPrivate": True, "enableGpu": True, "enableInternet": False,
        "competitionDataSources": ["enveda-CASMI26-molecule-id-mass-spectra"],
        "datasetDataSources": ["sanpier/casmi-spectrum-model-v1", "dmitriigluzdov/casmi26-fold-safe-fpnet",
                               "dmitriigluzdov/casmi26-natural-product-knowledge-table", "dmitriigluzdov/casmi26-natural-product-panel-benchmark",
                               "dmitriigluzdov/casmi26-pubchem-popularity-prior", "dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets",
                               "sanpier/catalogue-experiment-saner", "metric/rdkit-2026-3-3-wheel"]
                              + (["xaxipirulazo/casmi26-class3-data"] if a.val3 else []),
        "kernelDataSources": ["bgunshui/casmi26-scorer-train-public", "xaxipirulazo/casmi26-train-v2", "xaxipirulazo/casmi26-train-v3"]
                             + (["xaxipirulazo/casmi26-e180"] if a.ext else []) + a.extra_src,
    })
    print(r.status_code, r.text[:400])


if __name__ == "__main__":
    main()
