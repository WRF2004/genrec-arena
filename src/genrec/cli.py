import argparse
import csv
import json
from pathlib import Path

from .data import convert, prepare, synthetic, timestamp
from .engine import baselines, evaluate, load_model, predict, train


def main():
    p = argparse.ArgumentParser(description="GenRec Arena reproducible experiment CLI")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("demo-data")
    s.add_argument("--output", default="data/demo.csv")
    s.add_argument("--users", type=int, default=96)
    c = sub.add_parser("convert")
    c.add_argument("--kind", choices=["hm", "taobao", "canonical"], required=True)
    c.add_argument("--input", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--fraction", type=float, default=1.0)
    c.add_argument("--event", default="pv")
    d = sub.add_parser("prepare")
    d.add_argument("--input", required=True)
    d.add_argument("--output", required=True)
    d.add_argument("--train-end", required=True)
    d.add_argument("--val-end", required=True)
    d.add_argument("--max-history", type=int, default=30)
    d.add_argument("--feature-dim", type=int, default=64)
    d.add_argument("--metadata")
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--run", required=True)
    t.add_argument("--config", default="configs/demo.json")
    t.add_argument("--mode", choices=["generative", "sequence"], default="generative")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--device", default="cpu")
    e = sub.add_parser("evaluate")
    e.add_argument("--data", required=True)
    e.add_argument("--run", required=True)
    e.add_argument("--split", choices=["val", "test"], default="test")
    e.add_argument("--device", default="cpu")
    b = sub.add_parser("baselines")
    b.add_argument("--data", required=True)
    b.add_argument("--output", default="runs/baselines.json")
    b.add_argument("--k", type=int, default=12)
    b.add_argument("--split", choices=["val", "test"], default="test")
    r = sub.add_parser("recommend")
    r.add_argument("--data", required=True)
    r.add_argument("--run", required=True)
    r.add_argument("--history", default="", help="comma-separated raw item IDs")
    r.add_argument("--device", default="cpu")
    r.add_argument("--k", type=int, default=12)
    r.add_argument("--beam", type=int, default=32)
    h = sub.add_parser("hm-submit")
    h.add_argument("--data", required=True)
    h.add_argument("--run", required=True)
    h.add_argument("--transactions", required=True)
    h.add_argument("--sample", required=True)
    h.add_argument("--output", default="runs/submission.csv")
    h.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.command == "demo-data":
        synthetic(args.output, users=args.users)
        print(args.output)
    elif args.command == "convert":
        print(convert(args.input, args.output, args.kind, args.fraction, args.event))
    elif args.command == "prepare":
        print(
            json.dumps(
                prepare(
                    args.input,
                    args.output,
                    timestamp(args.train_end),
                    timestamp(args.val_end),
                    args.max_history,
                    args.feature_dim,
                    args.metadata,
                ),
                indent=2,
            )
        )
    elif args.command == "train":
        train(args.data, args.run, args.config, args.mode, args.resume, args.device)
    elif args.command == "evaluate":
        print(
            json.dumps(evaluate(args.data, args.run, args.split, args.device), indent=2)
        )
    elif args.command == "baselines":
        print(
            json.dumps(baselines(args.data, args.output, args.k, args.split), indent=2)
        )
    elif args.command == "recommend":
        model, index, state = load_model(args.run, args.device)
        items = json.loads((Path(args.data) / "items.json").read_text())
        lookup = {v: i for i, v in enumerate(items)}
        from .engine import fingerprint

        if fingerprint(args.data) != state["data_hash"]:
            raise ValueError("dataset does not match checkpoint")
        unknown = [i for i in args.history.split(",") if i and i not in lookup]
        if unknown:
            raise ValueError(
                f"unknown history items: {unknown}; use empty history for cold start"
            )
        hist = [lookup[i] for i in args.history.split(",") if i]
        _, rows = predict(
            model,
            [{"user": "query", "history": hist, "targets": [-1]}],
            state["mode"],
            index,
            state["config"],
            args.k,
            args.beam,
        )
        print(json.dumps([items[i] for i in rows[0]["predictions"]]))
    elif args.command == "hm-submit":
        submission(args)


def submission(args):
    """Inference export, preserving sample order and leading zeros. No leaderboard upload."""
    from collections import defaultdict

    import torch

    from .engine import fingerprint

    model, index, state = load_model(args.run, args.device)
    if fingerprint(args.data) != state["data_hash"]:
        raise ValueError("dataset does not match checkpoint")
    torch.set_num_threads(state["config"].get("threads", 2))
    items = json.loads((Path(args.data) / "items.json").read_text())
    if len(items) < 12:
        raise ValueError("H&M submission requires at least 12 catalog items")
    lookup = {v: i for i, v in enumerate(items)}
    histories = defaultdict(list)
    with open(args.transactions, newline="", encoding="utf-8") as f:
        for n, row in enumerate(csv.DictReader(f)):
            if row["article_id"] in lookup:
                histories[row["customer_id"]].append(
                    (row["t_dat"], n, lookup[row["article_id"]])
                )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with (
        open(args.sample, newline="", encoding="utf-8") as src,
        open(args.output, "w", newline="", encoding="utf-8") as dst,
    ):
        writer = csv.DictWriter(dst, fieldnames=["customer_id", "prediction"])
        writer.writeheader()
        for row in csv.DictReader(src):
            user = row["customer_id"]
            history = [x[2] for x in sorted(histories.get(user, []))][
                -state["config"]["max_history"] :
            ]
            _, pred = predict(
                model,
                [{"user": user, "history": history, "targets": [-1]}],
                state["mode"],
                index,
                state["config"],
                12,
                max(12, state["config"]["beam"]),
            )
            writer.writerow(
                {
                    "customer_id": user,
                    "prediction": " ".join(items[i] for i in pred[0]["predictions"]),
                }
            )
    print(args.output)


if __name__ == "__main__":
    main()
