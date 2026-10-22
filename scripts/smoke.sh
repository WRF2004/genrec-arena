#!/usr/bin/env bash
set -euo pipefail
python -m genrec.cli demo-data
python -m genrec.cli prepare --input data/demo.csv --output data/demo --train-end 16000 --val-end 20000 --max-history 12
python -m genrec.cli baselines --data data/demo --output runs/demo-baselines.json
python -m genrec.cli train --data data/demo --run runs/demo-gen --config configs/demo.json
python -m genrec.cli evaluate --data data/demo --run runs/demo-gen
python -m genrec.cli train --data data/demo --run runs/demo-seq --config configs/demo.json --mode sequence
python -m genrec.cli evaluate --data data/demo --run runs/demo-seq
