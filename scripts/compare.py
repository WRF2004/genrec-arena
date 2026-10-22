"""Paired user bootstrap; reports metric(A)-metric(B)."""

import argparse
import json
from pathlib import Path

import numpy as np

from genrec.data import dump
from genrec.metrics import metrics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--a", required=True)
p.add_argument("--b", required=True)
p.add_argument("--metric", choices=["MAP", "NDCG", "Recall", "Hit"], default="MAP")
p.add_argument("--k", type=int, default=12)
p.add_argument("--output", default="runs/comparison.json")
p.add_argument("--samples", type=int, default=2000)
a = p.parse_args()
xrows, yrows = [json.loads(Path(path).read_text()) for path in (a.a, a.b)]
x, y = {r["user"]: r for r in xrows}, {r["user"]: r for r in yrows}
if not x or x.keys() != y.keys() or len(x) != len(xrows) or len(y) != len(yrows):
    raise ValueError("same unique users required")
key = f"{a.metric}@{a.k}"
diffs = []
for user in sorted(x):
    if sorted(x[user]["targets"]) != sorted(y[user]["targets"]):
        raise ValueError("different target labels")
    diffs.append(
        metrics(x[user]["predictions"], x[user]["targets"], a.k)[key]
        - metrics(y[user]["predictions"], y[user]["targets"], a.k)[key]
    )
rng = np.random.default_rng(42)
diffs = np.array(diffs)
means = [
    float(rng.choice(diffs, len(diffs), replace=True).mean()) for _ in range(a.samples)
]
result = {
    "metric": key,
    "users": len(diffs),
    "difference_A_minus_B": float(diffs.mean()),
    "ci95": np.percentile(means, [2.5, 97.5]).tolist(),
    "bootstrap_samples": a.samples,
    "seed": 42,
}
dump(a.output, result)
print(json.dumps(result, indent=2))
