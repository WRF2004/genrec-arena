import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from .io import read_json, write_json, fingerprint


def main():
    p = argparse.ArgumentParser(description="DualSID Qwen reproducible recommendation experiments")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo"); d.add_argument("--output", required=True)
    d = sub.add_parser("prepare")
    for key in ("events", "items", "output"):
        d.add_argument("--" + key, required=True)
    d.add_argument("--train-end", type=float, required=True)
    d.add_argument("--val-end", type=float, required=True)
    d.add_argument("--max-history", type=int, default=30)
    d = sub.add_parser("sid"); d.add_argument("--data", required=True); d.add_argument("--output", required=True)
    d.add_argument("--levels", type=int, default=3); d.add_argument("--size", type=int, default=64)
    d.add_argument("--dim", type=int, default=64); d.add_argument("--seed", type=int, default=42)
    d.add_argument("--embeddings", help=".npz with item_ids and embeddings; reordered to frozen catalog")
    d = sub.add_parser("encode")
    d.add_argument("--data", required=True); d.add_argument("--output", required=True)
    d.add_argument("--model", default="Qwen/Qwen2.5-1.5B"); d.add_argument("--device", default="cuda")
    d.add_argument("--batch-size", type=int, default=8); d.add_argument("--max-length", type=int, default=128)
    d = sub.add_parser("train")
    for key in ("data", "sid", "config", "output"):
        d.add_argument("--" + key, required=True)
    d.add_argument("--resume", help="Exact Trainer checkpoint directory")
    d = sub.add_parser("baseline"); d.add_argument("--data", required=True)
    d.add_argument("--output", required=True); d.add_argument("--k", type=int, default=10)
    d.add_argument("--split", choices=["val", "test"], default="test")
    for command in ("evaluate", "recommend", "tune"):
        d = sub.add_parser(command)
        d.add_argument("--run", required=True); d.add_argument("--device", default="cpu")
        d.add_argument("--k", type=int, default=10); d.add_argument("--beam", type=int, default=20)
        d.add_argument("--mode", choices=["dual", "semantic", "collaborative"], default="dual")
        d.add_argument("--exclude-seen", action="store_true")
        d.add_argument("--output", required=True)
        if command != "tune":
            d.add_argument("--alpha", type=float, default=0.5)
        if command != "recommend":
            d.add_argument("--data", required=True)
        if command == "evaluate":
            d.add_argument("--split", choices=["val", "test"], default="test")
        if command == "recommend":
            d.add_argument("--history", default="", help="Comma-separated raw item IDs")
    a = p.parse_args()
    if a.command == "demo":
        from .data import demo
        data = demo(a.output)
        print({s: len(r) for s, r in data["splits"].items()})
    elif a.command == "prepare":
        from .data import prepare
        prepare(a.events, a.items, a.output, a.train_end, a.val_end, a.max_history)
    elif a.command == "sid":
        from .sid import fit
        data = read_json(a.data); embeddings = None
        if a.embeddings:
            z = np.load(a.embeddings, allow_pickle=False)
            ids = [str(i) for i in z["item_ids"]]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate embedding item IDs")
            lookup = dict(zip(ids, z["embeddings"]))
            embeddings = np.stack([lookup[i] for i in data["catalog"]])
        artifact = fit(data, a.levels, a.size, a.dim, a.seed, embeddings)
        write_json(a.output, artifact)
        print(artifact["diagnostics"])
    elif a.command == "encode":
        from .encode import encode
        encode(read_json(a.data), a.output, a.model, a.device, a.batch_size, a.max_length)
    elif a.command == "train":
        from .train import train
        train(read_json(a.data), read_json(a.sid), read_json(a.config), a.output, a.resume)
    elif a.command == "baseline":
        from .retrieval import ranking_metrics
        data = read_json(a.data)
        counts = Counter(i for seq in data["train_sequences"] for i in seq)
        ranked = sorted(counts, key=lambda i: (-counts[i], i))[:a.k]
        rows = data["splits"][a.split]
        write_json(a.output, ranking_metrics([ranked] * len(rows), [r["targets"] for r in rows], data["catalog"], a.k))
    else:
        from .train import load_run
        from .retrieval import evaluate, recommend
        model, tok, sid, cfg = load_run(a.run, a.device)
        if cfg.get("mode", "dual") != "dual" and a.mode != cfg["mode"]:
            raise ValueError("Single-stream checkpoint cannot evaluate an untrained stream")
        if a.command == "recommend":
            result = recommend(model, tok, sid, [i for i in a.history.split(",") if i], cfg["max_length"],
                               a.k, a.beam, a.alpha, a.mode, a.exclude_seen)
        else:
            data = read_json(a.data)
            if fingerprint(data) != sid["data_hash"]:
                raise ValueError("Evaluation data differs from training snapshot")
            if a.command == "tune":
                if a.mode != "dual":
                    raise ValueError("tune is for dual-stream fusion")
                trials = [evaluate(model, tok, sid, data, "val", cfg, a.k, a.beam,
                                   alpha, "dual", a.exclude_seen)["metrics"] for alpha in (0., .25, .5, .75, 1.)]
                best = max(trials, key=lambda m: m[f"ndcg@{a.k}"])
                result = {"best_alpha": best["alpha"], "selection_split": "val", "trials": trials}
            else:
                if cfg.get("mode", "dual") != "dual" and a.mode != cfg["mode"]:
                    raise ValueError("Single-stream checkpoint cannot evaluate an untrained stream")
                result = evaluate(model, tok, sid, data, a.split, cfg, a.k, a.beam,
                                  a.alpha, a.mode, a.exclude_seen)
        write_json(a.output, result)
        print(result.get("metrics", result) if isinstance(result, dict) else result)


if __name__ == "__main__":
    main()
