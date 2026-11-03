#!/usr/bin/env bash
# One-shot setup of a fresh GPU box (B200/H100) for fast iteration.
#   KAGGLE_API_TOKEN=... bash scripts/setup_b200.sh /data
# Steps are idempotent (skipped when their output exists). ~1 h end to end, the
# 18M-candidate filter + embeddings dominate; afterwards every experiment only
# needs tools/sweep.py.
set -euo pipefail
BASE=${1:-/data}
D=$BASE/data; W=$BASE/work; R=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$D/cands" "$W"
py() { python3 "$R/scripts/$1.py" "${@:2}"; }

# 1. environment (PyTorch for the box's CUDA must already be installed)
pip install -q -r "$R/requirements.txt" kaggle

# 2. data: competition (CC BY-NC, never commit) + public candidate datasets
C=enveda-CASMI26-molecule-id-mass-spectra
[ -f "$D/train.parquet" ] || { kaggle competitions download -c $C -p "$D" && (cd "$D" && unzip -oq "$C.zip" && rm "$C.zip"); }
[ -d "$D/cands/rishav" ] || kaggle datasets download rishavae/casmi-2026-candidate-database -p "$D/cands/rishav" --unzip
[ -d "$D/cands/pcl" ]    || kaggle datasets download xjoannax88/casmi-pubchemlite-candidates -p "$D/cands/pcl" --unzip
COCO=$(find "$D/cands/rishav" -name 'coconut_pure_shard_*.parquet' | head -1)
PUB=$(dirname "$(find "$D/cands/rishav" -name 'pubchem_pure_shard_01.parquet' | head -1)")/'*.parquet'
PCL=$(find "$D/cands/pcl" -name 'pubchemlite_structures.parquet' | head -1)

# 3. preprocessing (CPU)
[ -f "$W/lib/peaks.npz" ]          || py prep_library --train "$D/train.parquet" --out "$W/lib"
[ -f "$W/candidates.parquet" ]     || py build_candidates --coconut "$COCO" --pubchem "$PUB" --pcl "$PCL" --out "$W/candidates.parquet"
[ -f "$W/holdout.parquet" ]        || py make_split --lib "$W/lib" --db "$W/candidates.parquet" --out "$W"

# 4. GPU features: PLM embeddings (centred), SAE, JEPA encoder
[ -f "$W/emb/plm.npy" ]            || py embed_molecules --work "$W" --out "$W/emb"
[ -f "$W/emb/plm_mean.npy" ]       || py center_plm --emb "$W/emb"
[ -f "$W/runs/sae.pt" ]            || py train_sae --work "$W" --out "$W/runs/sae.pt"
[ -f "$W/runs/jepa/encoder.pt" ]   || py pretrain_jepa --work "$W" --out "$W/runs/jepa" --steps 30000 --batch 2048 --lr 1e-3

# 5. throughput check
python3 "$R/tools/sweep.py" "$R/exp/b200_bench.json" --work "$W"
python3 "$R/tools/report.py" "$W/results.jsonl" --filter bench/
