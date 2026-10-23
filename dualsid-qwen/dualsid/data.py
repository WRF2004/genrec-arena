"""Strict global temporal windows; no future interactions in feature fitting."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from .io import write_json


def prepare(events_path, items_path, output, train_end, val_end, max_history=30):
    if train_end >= val_end or max_history < 1:
        raise ValueError("Require train_end < val_end and positive max_history")
    items = {}
    with open(items_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["item_id"] in items:
                raise ValueError("Duplicate item_id in metadata")
            items[r["item_id"]] = r["text"]
    users = defaultdict(list)
    with open(events_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            t = float(r["timestamp"])
            if not np.isfinite(t):
                raise ValueError("Non-finite timestamp")
            users[r["user_id"]].append((t, r["item_id"]))
    # Exact duplicate events do not create artificial sequence positions.
    users = {u: sorted(set(rows)) for u, rows in sorted(users.items())}
    catalog = sorted({i for rows in users.values() for t, i in rows if t <= train_end})
    if len(catalog) < 2:
        raise ValueError("At least two training items required")
    if set(catalog) - items.keys():
        raise ValueError("Training item missing metadata")
    splits = {s: [] for s in ("train", "val", "test")}
    train_sequences = []
    for u, rows in users.items():
        train = [(t, i) for t, i in rows if t <= train_end]
        if train:
            train_sequences.append([i for _, i in train])
        # Equal-timestamp actions cannot see one another.
        for t, item in train:
            history = [i for ht, i in train if ht < t][-max_history:]
            if history:
                splits["train"].append({"user": u, "history": history, "target": item})
        for split, low, high, history_end in (
            ("val", train_end, val_end, train_end),
            ("test", val_end, float("inf"), val_end),
        ):
            targets = [(t, i) for t, i in rows if low < t <= high]
            if not targets:
                continue
            latest = max(t for t, _ in targets)
            # One deterministic target/user/window; tied last targets form one relevance set.
            relevant = sorted({i for t, i in targets if t == latest})
            history = [i for t, i in rows if t <= history_end][-max_history:]
            splits[split].append({"user": u, "history": history, "targets": relevant})
    if not all(splits.values()):
        raise ValueError("Train/val/test must all contain examples; adjust cutoffs")
    result = dict(catalog=catalog, texts=[items[i] for i in catalog],
                  train_sequences=train_sequences, splits=splits,
                  protocol=dict(train_end=train_end, val_end=val_end, max_history=max_history,
                                catalog="train-only", evaluation="last timestamp per window"))
    write_json(Path(output) / "dataset.json", result)
    return result


def demo(output, seed=42):
    """Synthetic categories with repeated preferences. Not a public benchmark."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with (output / "items.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["item_id", "text"])
        for i in range(32):
            w.writerow([f"i{i:03}", f"category {i // 8} product {i}"])
    with (output / "events.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["user_id", "item_id", "timestamp"])
        for u in range(24):
            for t in range(1, 13):
                item = (u % 4) * 8 + int(rng.integers(8))
                w.writerow([f"u{u}", f"i{item:03}", t])
    return prepare(output / "events.csv", output / "items.csv", output, 8, 10, 8)
