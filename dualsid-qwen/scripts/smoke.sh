#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
root="${1:-runs/smoke}"
if [ -e "$root" ]; then
  echo "Choose a fresh output path: $root exists" >&2
  exit 1
fi
python -m dualsid.cli demo --output "$root/data"
python -m dualsid.cli sid --data "$root/data/dataset.json" --output "$root/sid.json" --levels 2 --size 4 --dim 8
python -m dualsid.cli train --data "$root/data/dataset.json" --sid "$root/sid.json" --config configs/tiny.json --output "$root/model"
python -m dualsid.cli baseline --data "$root/data/dataset.json" --output "$root/popularity.json" --k 5
python -m dualsid.cli tune --run "$root/model" --data "$root/data/dataset.json" --output "$root/fusion.json" --k 5 --beam 8
alpha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_alpha"])' "$root/fusion.json")
python -m dualsid.cli evaluate --run "$root/model" --data "$root/data/dataset.json" --alpha "$alpha" --k 5 --beam 8 --output "$root/test.json"
python -m dualsid.cli recommend --run "$root/model" --history i000,i001 --k 5 --beam 8 --output "$root/recommendation.json"
