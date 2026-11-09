#!/usr/bin/env bash
# Download a dataset file by file, keeping its folder structure (for datasets whose whole-zip download 404s).
#   bash tools/fetch_files.sh owner/slug /kaggle/input/slug
set -u; d=$1; out=$2; K=${KAGGLE_BIN:-kaggle}; tok=""
while :; do
  page=$($K datasets files $d --page-size 200 ${tok:+--page-token $tok} 2>&1)
  echo "$page" | grep -v "Next Page" | awk 'NR>2{print $1}' >> $out.files.tmp
  tok=$(echo "$page" | grep -oP "Next Page Token = \K\S+"); [ -z "$tok" ] && break
done
sort -u $out.files.tmp > $out.files; rm -f $out.files.tmp
while read -r f; do
  dir=$out/$(dirname "$f"); [ -f "$out/$f" ] && continue; mkdir -p "$dir"
  $K datasets download -d $d -f "$f" -p "$dir" -q >/dev/null 2>&1 || echo "FAILED $f"
  [ -f "$dir/$(basename "$f").zip" ] && (cd "$dir" && unzip -q -o "$(basename "$f").zip" && rm -f "$(basename "$f").zip")
done < $out.files
echo "done $d: $(find $out -type f | wc -l) files"
