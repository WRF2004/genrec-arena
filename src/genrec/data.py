"""Strict global-time split; catalog and learned statistics use train only."""

import csv
import hashlib
import json
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def timestamp(value):
    try:
        return float(value)
    except ValueError:
        dt = datetime.fromisoformat(value)
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def keep_user(user, fraction):
    return int(hashlib.sha256(user.encode()).hexdigest()[:12], 16) / 16**12 < fraction


def convert(source, output, kind, fraction=1.0, event="pv"):
    """Streaming user-hash sample, preserving all selected users' histories."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        open(source, newline="", encoding="utf-8") as src,
        open(output, "w", newline="", encoding="utf-8") as dst,
    ):
        writer = csv.DictWriter(
            dst, fieldnames=["user_id", "item_id", "timestamp", "category"]
        )
        writer.writeheader()
        rows = csv.reader(src) if kind == "taobao" else csv.DictReader(src)
        for row in rows:
            if kind == "taobao":
                user, item, cat, behavior, ts = row
                if behavior != event:
                    continue
            elif kind == "hm":
                user, item, ts, cat = (
                    row["customer_id"],
                    row["article_id"],
                    row["t_dat"],
                    "",
                )
            else:
                user, item, ts, cat = (
                    row["user_id"],
                    row["item_id"],
                    row["timestamp"],
                    row.get("category", ""),
                )
            if keep_user(user, fraction):
                writer.writerow(
                    {
                        "user_id": user,
                        "item_id": item,
                        "timestamp": timestamp(ts),
                        "category": cat,
                    }
                )
                count += 1
    return count


def synthetic(output, users=96, items=48, seed=42):
    """Intentionally learnable cyclic preferences. Pipeline test, NOT a benchmark."""
    rng = np.random.default_rng(seed)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "item_id", "timestamp", "category"])
        for u in range(users):
            group = u % 6
            pool = list(range(group, items, 6))
            offset = int(rng.integers(len(pool)))
            for t in range(24):
                item = pool[(t + offset) % len(pool)]
                if rng.random() < 0.08:
                    item = int(rng.integers(items))
                w.writerow([f"u{u}", f"i{item:04d}", t * 1000 + u, f"g{item % 6}"])


def feature_hash(text, dim):
    vector = np.zeros(dim, np.float32)
    # Stable feature hashing; accepts categorical fields and text without a fitted vocabulary.
    for token in text.lower().split():
        h = hashlib.sha256(token.encode()).digest()
        vector[int.from_bytes(h[:4], "little") % dim] += 1 if h[4] % 2 else -1
    norm = np.linalg.norm(vector)
    return vector / max(norm, 1e-12)


def prepare(
    source, output, train_end, val_end, max_history=30, feature_dim=64, metadata=None
):
    if not train_end < val_end or max_history < 1:
        raise ValueError("require train_end < val_end and positive max_history")
    histories = defaultdict(list)
    cats = {}
    hasher = hashlib.sha256()
    with open(source, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(block)
    sha = hasher.hexdigest()
    with open(source, newline="", encoding="utf-8") as f:
        for order, r in enumerate(csv.DictReader(f)):
            t = timestamp(r["timestamp"])
            if not np.isfinite(t) or not r["user_id"] or not r["item_id"]:
                raise ValueError(f"invalid interaction row {order + 2}")
            histories[r["user_id"]].append((t, order, r["item_id"]))
            if t < train_end:
                cats.setdefault(r["item_id"], r.get("category", ""))
    ids = sorted(cats)
    if not ids:
        raise ValueError("no training catalog; check timestamp units/cutoffs")
    item_map = {v: i for i, v in enumerate(ids)}
    features = {i: f"category:{cats[i]}" if cats[i] else "unknown" for i in ids}
    if metadata:
        # metadata must be a snapshot available before train_end; no future engagement features.
        with open(metadata, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = r.get("item_id", r.get("article_id"))
                if key in features:
                    features[key] = " ".join(
                        f"{k}:{v}"
                        for k, v in r.items()
                        if k not in ("item_id", "article_id") and v
                    )
    splits = {"train": [], "val": [], "test": []}
    for user, events in sorted(histories.items()):
        events.sort()
        train = [(t, i) for t, _, i in events if t < train_end]
        train_times = [t for t, _ in train]
        train_items = [item_map[i] for _, i in train]
        for j in range(1, len(train)):
            # Same-timestamp items are not assumed to have an observable order.
            stop = bisect_left(train_times, train[j][0])
            context = train_items[max(0, stop - max_history) : stop]
            if context:
                splits["train"].append(
                    {
                        "user": user,
                        "history": context,
                        "targets": [item_map[train[j][1]]],
                    }
                )
        for name, lower, upper in [
            ("val", train_end, val_end),
            ("test", val_end, float("inf")),
        ]:
            # Test uses observed validation-window history, but no parameter fitting on it.
            past = [item_map[i] for t, _, i in events if t < lower and i in item_map][
                -max_history:
            ]
            raw_targets = sorted({i for t, _, i in events if lower <= t < upper})
            if raw_targets:
                splits[name].append(
                    {
                        "user": user,
                        "history": past,
                        "targets": [item_map.get(i, -1) for i in raw_targets],
                        "raw_targets": raw_targets,
                    }
                )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.save(
        out / "features.npy",
        np.stack([feature_hash(features[i], feature_dim) for i in ids]),
    )
    dump(out / "items.json", ids)
    for name in ("train", "val", "test"):
        if not splits[name]:
            raise ValueError(f"empty {name} split; choose different cutoffs")
    stats = {
        "sha256": sha,
        "train_end": train_end,
        "val_end": val_end,
        "catalog_size": len(ids),
        "users": len(histories),
        "max_history": max_history,
        "protocol": "global time windows; frozen train catalog; all window targets; repeat purchases allowed",
        "rows": {k: len(v) for k, v in splits.items()},
        "cold_target_fraction": {
            k: sum(t == -1 for r in splits[k] for t in r["targets"])
            / sum(len(r["targets"]) for r in splits[k])
            for k in ("val", "test")
        },
    }
    dump(out / "manifest.json", stats)
    return stats


def load_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
