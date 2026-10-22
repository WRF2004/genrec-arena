import math

import numpy as np


def metrics(predictions, targets, k):
    """Unreachable cold items (-1) retain their multiplicity in the denominator."""
    if not targets:
        raise ValueError("empty relevance set")
    known = {t for t in targets if t >= 0}
    total = len(known) + sum(t < 0 for t in targets)
    hits, ap, dcg, seen = 0, 0.0, 0.0, set()
    for rank, item in enumerate(predictions[:k], 1):
        if item in known and item not in seen:
            hits += 1
            ap += hits / rank
            dcg += 1 / math.log2(rank + 1)
        seen.add(item)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(total, k)))
    return {
        f"Recall@{k}": hits / total,
        f"Hit@{k}": float(hits > 0),
        f"NDCG@{k}": dcg / ideal,
        f"MAP@{k}": ap / min(total, k),
    }


def aggregate(rows, k):
    if not rows:
        raise ValueError("no evaluation users")
    values = [metrics(r["predictions"], r["targets"], k) for r in rows]
    result = {key: float(np.mean([v[key] for v in values])) for key in values[0]}
    result["users"] = len(rows)
    return result
