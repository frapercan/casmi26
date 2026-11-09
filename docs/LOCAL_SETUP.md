# Running the project on a local machine

The project was developed in a cloud container that could only reach Kaggle through an API proxy. Outputs of
Kaggle runs could not be downloaded, so every analysis ran as a Kaggle kernel. On a local machine with a GPU,
the frozen public pipeline and our validations can run here, without the weekly Kaggle GPU quota.

## 1. Prerequisites

- Linux, or WSL2 (Ubuntu) on Windows, with an NVIDIA driver (`nvidia-smi` must work).
- RAM: 32 GB recommended. Kaggle sessions have about 30 GB, and the pipeline plus the spectral library are sized
  for that.
- Disk: roughly 50–100 GB free for the competition data, the pipeline datasets and their outputs (an estimate).
- Python 3.11 or later. Kaggle currently runs Python 3.13.

## 2. Python environment

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt requests jupyter nbconvert kaggle
pip install torch   # pick the CUDA build for your driver at pytorch.org
```

`scripts/train_jax.py` (the TPU trainer) also needs `jax`, `flax` and `optax`. It is only needed for training
in JAX.

## 3. Kaggle credentials

Create an API token on Kaggle (Settings → API) and export it in your shell profile:

```bash
export KAGGLE_API_TOKEN=...   # never commit it, never paste it into a chat
```

Every script that calls the Kaggle API reads `KAGGLE_API_TOKEN`, then `~/.kaggle/access_token` (the file the
`kaggle` CLI uses), then falls back to the cloud proxy. The scripts send a fixed User-Agent: the API answers
404 to the default `python-requests/2.34.x` agent.

## 4. Data: stays on this machine

The competition data is CC BY-NC. It must not be committed or redistributed. `.gitignore` already excludes
`data/`, `work/`, `splits/val3/`, `*.parquet`, `*.csv` and `*.npz`.

Mirror Kaggle's `/kaggle` layout so the frozen notebooks run unmodified. The notebooks find most inputs by
recursive search under `/kaggle/input`, so each source just needs its own folder named after its slug.

Keep the data in your home and link it as `/kaggle` (the one step that needs sudo):

```bash
mkdir -p ~/casmi/kaggle/input ~/casmi/kaggle/working && sudo ln -sfn ~/casmi/kaggle /kaggle
```

`tools/fetch_inputs.sh` downloads everything below (idempotent; `KROOT` sets the target root).

```bash
C=enveda-CASMI26-molecule-id-mass-spectra
kaggle competitions download -c $C -p /kaggle/input/competitions/$C && (cd /kaggle/input/competitions/$C && unzip -q -o ./*.zip)
```

```bash
for d in sanpier/casmi-spectrum-model-v1 dmitriigluzdov/casmi26-fold-safe-fpnet \
         dmitriigluzdov/casmi26-natural-product-knowledge-table dmitriigluzdov/casmi26-natural-product-panel-benchmark \
         dmitriigluzdov/casmi26-pubchem-popularity-prior dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets \
         sanpier/catalogue-experiment-saner metric/rdkit-2026-3-3-wheel \
         samartalwar/casmi-2026-spectral-library-massbankharmonized takumuhata/casmi26-extlib \
         qwertykeypadapple/casmi26-lotus-candidates \
         xaxipirulazo/casmi26-speclib xaxipirulazo/casmi26-class3-data; do
  kaggle datasets download -d $d -p /kaggle/input/$(basename $d) --unzip
done
```

```bash
for k in bgunshui/casmi26-scorer-train-public xaxipirulazo/casmi26-train-v2 xaxipirulazo/casmi26-train-wm7c3 \
         xaxipirulazo/casmi26-eval-c3; do
  kaggle kernels output $k -p /kaggle/input/$(basename $k)
done
```

The first group of datasets is what the public 0.480 pipeline uses. The ExtLib/MassBank/LOTUS ones are used by
the public 0.484 ExtLib notebook. The `xaxipirulazo/*` sources are our private datasets and kernel outputs:
the training-spectra library, the class-3 validation cell, our trained models and the class-3 evaluation outputs.

## 5. The public notebooks we build on

- **0.480 pipeline (frozen, SHA-pinned):** `kaggle kernels pull xaxipirulazo/casmi26-hybrid-w0 -p nb/`, then drop
  the last cell. `hybrid/build.py` appended exactly one cell to the public notebook.
- **0.484 ExtLib:** `kaggle kernels pull alexchilton/casmi26-lb-0-484-peaks-to-structures-extlib -p nb/`.

The builders in `hybrid/` that take `--dry out.ipynb` write the full notebook (pipeline plus our cells)
instead of pushing it. To run one locally:

```bash
cd /kaggle/working && jupyter nbconvert --to notebook --execute /path/out.ipynb --ExecutePreprocessor.timeout=-1 --output-dir /kaggle/working
```

## 6. What still has to run on Kaggle

The hidden test only exists inside Kaggle's scoring rerun. Final submissions and probe submissions run there,
on Kaggle hardware (4 CPU cores, a T4 or P100 GPU, about 30 GB RAM), in under 9 hours. Anything developed
locally has to fit those limits.

## 7. Helper

`tools/kaggle_api.py`: kernel status, waiting on and reading kernel logs, submissions, the leaderboard with the
medal lines, and private dataset uploads. Run it without arguments for usage.
