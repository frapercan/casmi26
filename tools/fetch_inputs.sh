#!/usr/bin/env bash
# Mirror the Kaggle /kaggle/input layout locally (data stays on this machine; never commit it).
#   KROOT=/home/xaxi/casmi/kaggle bash tools/fetch_inputs.sh
set -u
KROOT=${KROOT:-/kaggle}; IN=$KROOT/input; K=${KAGGLE_BIN:-kaggle}
C=enveda-CASMI26-molecule-id-mass-spectra
mkdir -p $IN/competitions/$C
[ -f $IN/competitions/$C/train.parquet ] || { $K competitions download -c $C -p $IN/competitions/$C && (cd $IN/competitions/$C && unzip -q -o ./*.zip && rm -f ./*.zip); }
for d in sanpier/casmi-spectrum-model-v1 dmitriigluzdov/casmi26-fold-safe-fpnet \
         dmitriigluzdov/casmi26-natural-product-knowledge-table dmitriigluzdov/casmi26-natural-product-panel-benchmark \
         dmitriigluzdov/casmi26-pubchem-popularity-prior dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets \
         sanpier/catalogue-experiment-saner metric/rdkit-2026-3-3-wheel \
         samartalwar/casmi-2026-spectral-library-massbankharmonized takumuhata/casmi26-extlib \
         qwertykeypadapple/casmi26-lotus-candidates xaxipirulazo/casmi26-speclib xaxipirulazo/casmi26-class3-data; do
  out=$IN/$(basename $d)
  [ -d $out ] && [ -n "$(ls -A $out 2>/dev/null)" ] && { echo "have $d"; continue; }
  echo "== $d"; $K datasets download -d $d -p $out --unzip || echo "FAILED $d"
done
for k in xaxipirulazo/casmi26-train-v2 xaxipirulazo/casmi26-train-wm7c3 xaxipirulazo/casmi26-eval-c3; do
  out=$IN/$(basename $k)
  [ -d $out ] && [ -n "$(ls -A $out 2>/dev/null)" ] && { echo "have $k"; continue; }
  echo "== kernel output $k"; mkdir -p $out; $K kernels output $k -p $out || echo "FAILED $k"
done
du -sh $IN/* | sort -h
