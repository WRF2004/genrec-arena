"""Run tokenizer ablations with identical seed/data/model settings; no test tuning."""

import argparse
import json
from pathlib import Path

from genrec.engine import evaluate, train

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--data", required=True)
p.add_argument("--config", default="configs/demo.json")
p.add_argument("--output", default="runs/ablations")
p.add_argument("--device", default="cpu")
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
base = json.loads(Path(a.config).read_text())
results = {}
for method in ("random", "kmeans", "rqvae"):
    cfg = {**base, "tokenizer": method}
    path = root / f"{method}.json"
    path.write_text(json.dumps(cfg, indent=2))
    train(a.data, root / method, path, device=a.device)
    results[method] = evaluate(a.data, root / method, split="val", device=a.device)
(root / "validation_comparison.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
