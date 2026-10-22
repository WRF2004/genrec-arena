import math
from collections import Counter, defaultdict


class ItemBaselines:
    """Sparse user co-occurrence cosine; retained top neighbors bound inference cost."""

    def __init__(self, train_rows, n_items, neighbors=100):
        users = defaultdict(set)
        self.pop = Counter()
        for r in train_rows:
            self.pop.update(r["targets"])
            users[r["user"]].update(r["history"] + r["targets"])
        counts, co = Counter(), defaultdict(Counter)
        for items in users.values():
            # Unique observed items per user. O(sum_u |I_u|^2): use sampled users for large datasets.
            counts.update(items)
            for i in items:
                co[i].update(j for j in items if i != j)
        self.neighbors = {}
        for i, row in co.items():
            scores = {
                j: count / math.sqrt(counts[i] * counts[j]) for j, count in row.items()
            }
            self.neighbors[i] = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[
                :neighbors
            ]
        self.order = sorted(range(n_items), key=lambda i: (-self.pop[i], i))

    def predict(self, history, k, kind):
        if kind == "popular" or not history:
            return self.order[:k]
        score = Counter()
        for age, i in enumerate(reversed(history)):
            for j, sim in self.neighbors.get(i, []):
                score[j] += sim / (age + 1)
        ranked = sorted(score, key=lambda i: (-score[i], i))
        seen = set(ranked)
        return (ranked + [i for i in self.order if i not in seen])[:k]
