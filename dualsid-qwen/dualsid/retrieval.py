import math
import time
from collections import defaultdict
import numpy as np
import torch
from .model import path_ids, prompt_ids
from .sid import Trie


@torch.inference_mode()
def beam_search(model, prompt, paths, beam=20):
    """Full-vocabulary log probabilities, then restrict to valid catalog branches.

    Batched beams, equal depth. No KV cache yet; returned scores include all SID
    tokens, but not EOS (constant forced terminal in this fixed-length space).
    """
    if beam < 1:
        raise ValueError("beam must be positive")
    if not paths:
        return []
    trie = Trie(paths)
    device = next(model.parameters()).device
    hypotheses = [((), 0.0)]
    depth = len(next(iter(paths.values())))
    for _ in range(depth):
        x = torch.tensor([prompt + list(prefix) for prefix, _ in hypotheses], device=device)
        logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[:, -1].float()
        lp = logits.log_softmax(-1)
        candidates = []
        for b, (prefix, score) in enumerate(hypotheses):
            for t in trie.allowed(prefix):
                candidates.append((prefix + (t,), score + float(lp[b, t])))
        hypotheses = sorted(candidates, key=lambda p: (-p[1], p[0]))[:beam]
    return [(trie.items[prefix], score) for prefix, score in hypotheses]


def fuse(semantic, collaborative, alpha=0.5, constant=60):
    if not 0 <= alpha <= 1 or constant < 0:
        raise ValueError("Invalid RRF weights")
    scores = defaultdict(float)
    for candidates, weight in ((semantic, alpha), (collaborative, 1 - alpha)):
        if weight == 0:
            continue
        for rank, (item, _) in enumerate(candidates, 1):
            scores[item] += weight / (constant + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))


def recommend(model, tokenizer, sid, history, max_length, k=10, beam=20,
              alpha=0.5, mode="dual", exclude_seen=False):
    if k < 1 or beam < k:
        raise ValueError("Require beam >= k >= 1")
    model.eval()
    results = {}
    seen = set(history)
    for stream in {"dual": ("s", "c"), "semantic": ("s",), "collaborative": ("c",)}[mode]:
        all_paths = path_ids(tokenizer, sid, stream)
        prompt = prompt_ids(tokenizer, all_paths, history, stream,
                            max_length - len(next(iter(all_paths.values()))) - 1)
        paths = {i: p for i, p in all_paths.items() if not exclude_seen or i not in seen}
        results[stream] = beam_search(model, prompt, paths, beam)
    if mode == "dual":
        return fuse(results["s"], results["c"], alpha)[:k]
    return [i for i, _ in next(iter(results.values()))][:k]


def ranking_metrics(predictions, targets, catalog, k):
    if not predictions or len(predictions) != len(targets):
        raise ValueError("Nonempty aligned predictions/targets required")
    sums = defaultdict(float)
    recommended = set()
    for pred, truth in zip(predictions, targets):
        if not truth:
            raise ValueError("Empty relevance set")
        pred = list(dict.fromkeys(pred))[:k]
        truth = set(truth)
        hits = [int(i in truth) for i in pred]
        sums["hit"] += float(any(hits))
        sums["recall"] += sum(hits) / len(truth)
        dcg = sum(h / math.log2(r + 2) for r, h in enumerate(hits))
        ideal = sum(1 / math.log2(r + 2) for r in range(min(k, len(truth))))
        sums["ndcg"] += dcg / ideal
        sums["oov_target_rate"] += len(truth - set(catalog)) / len(truth)
        recommended.update(pred)
    n = len(predictions)
    return {**{f"{m}@{k}" if m != "oov_target_rate" else m: v / n for m, v in sums.items()},
            "coverage": len(recommended) / len(catalog), "users": n}


def evaluate(model, tokenizer, sid, data, split, config, k=10, beam=20, alpha=0.5,
             mode="dual", exclude_seen=False):
    predictions, times = [], []
    # One warm-up call excluded from timing.
    rows = data["splits"][split]
    recommend(model, tokenizer, sid, rows[0]["history"], config["max_length"], k, beam, alpha, mode, exclude_seen)
    for row in rows:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        predictions.append(recommend(model, tokenizer, sid, row["history"], config["max_length"],
                                     k, beam, alpha, mode, exclude_seen))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    targets = [r["targets"] for r in rows]
    metrics = ranking_metrics(predictions, targets, sid["catalog"], k)
    metrics.update(latency_p50_ms=float(np.percentile(times, 50)), latency_p95_ms=float(np.percentile(times, 95)),
                   split=split, alpha=alpha, mode=mode, beam=beam, exclude_seen=exclude_seen)
    return {"metrics": metrics, "predictions": [dict(user=r["user"], targets=r["targets"], items=p)
                                               for r, p in zip(rows, predictions)]}
