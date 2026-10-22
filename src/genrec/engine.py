import copy
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .baselines import ItemBaselines
from .data import dump, load_rows
from .metrics import aggregate
from .models import CausalTransformer
from .tokenizer import SemanticIndex, fit_rqvae, residual_kmeans


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fingerprint(data_dir):
    h = hashlib.sha256()
    for name in (
        "manifest.json",
        "items.json",
        "features.npy",
        "train.jsonl",
        "val.jsonl",
        "test.jsonl",
    ):
        with (Path(data_dir) / name).open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def examples(rows, mode, index, max_history):
    records = []
    for r in rows:
        if mode == "generative":
            context = index.history_tokens(r["history"], max_history)
            target = index.codes[r["targets"][0]].tolist()
            seq = context + target
            x = seq[:-1]
            y = [-100] * (len(context) - 1) + target
        else:
            x = [1] + [i + 3 for i in r["history"][-max_history:]]
            y = [-100] * (len(x) - 1) + [r["targets"][0] + 3]
        records.append((x, y))
    return records


def collate(rows):
    length = max(len(x) for x, _ in rows)
    x = torch.zeros(len(rows), length, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (xs, ys) in enumerate(rows):
        x[i, : len(xs)] = torch.tensor(xs)
        y[i, : len(ys)] = torch.tensor(ys)
    return x, y


def make_model(config, mode, index, n_items):
    vocab = index.vocab if mode == "generative" else n_items + 3
    length = (
        (config["max_history"] + 1) * index.depth + 2
        if mode == "generative"
        else config["max_history"] + 1
    )
    return CausalTransformer(
        vocab,
        length,
        config["dim"],
        config["heads"],
        config["layers"],
        config["dropout"],
    )


@torch.no_grad()
def predict(model, rows, mode, index, config, k, beam):
    model.eval()
    output, times = [], []
    device = next(model.parameters()).device
    for r in rows:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        if mode == "generative":
            pred = index.beam_search(
                model, r["history"], k, beam, config["max_history"]
            )
        else:
            ids = [1] + [i + 3 for i in r["history"][-config["max_history"] :]]
            logits = model(torch.tensor([ids], device=device))[0, -1, 3:]
            pred = logits.topk(min(k, len(logits))).indices.tolist()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        output.append({"user": r["user"], "targets": r["targets"], "predictions": pred})
    result = aggregate(output, k)
    result["latency_ms_p50"] = float(np.median(times))
    result["latency_ms_p95"] = float(np.percentile(times, 95))
    result["coverage"] = len({i for r in output for i in r["predictions"]}) / len(
        index.codes
    )
    return result, output


def train(
    data_dir, run_dir, config_path, mode="generative", resume=False, device="cpu"
):
    config = json.loads(Path(config_path).read_text())
    if config["epochs"] < 1 or config["beam"] < config["k"]:
        raise ValueError("epochs >=1 and beam >= k required")
    seed_all(config["seed"])
    torch.set_num_threads(config.get("threads", 2))
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    last_path = out / "last.pt"
    if last_path.exists() and not resume:
        raise ValueError("run already exists; use --resume or a different --run")
    train_rows, val_rows = (
        load_rows(Path(data_dir) / "train.jsonl"),
        load_rows(Path(data_dir) / "val.jsonl"),
    )
    features = np.load(Path(data_dir) / "features.npy")
    data_hash = fingerprint(data_dir)
    state = (
        torch.load(last_path, map_location="cpu", weights_only=False)
        if resume
        else None
    )
    if state:
        old = {k: v for k, v in state["config"].items() if k != "epochs"}
        new = {k: v for k, v in config.items() if k != "epochs"}
        if state["data_hash"] != data_hash or state["mode"] != mode or old != new:
            raise ValueError(
                "resume config/mode/data mismatch (only epochs may change)"
            )
        codes = np.array(state["raw_codes"])
    else:
        if config["tokenizer"] == "rqvae":
            codes, tokenizer_state, error = fit_rqvae(
                features,
                config["levels"],
                config["codebook_size"],
                config["seed"],
                config.get("tokenizer_epochs", 30),
                device,
            )
            torch.save(tokenizer_state, out / "rqvae.pt")
        elif config["tokenizer"] == "random":
            codes = np.random.default_rng(config["seed"]).integers(
                config["codebook_size"], size=(len(features), config["levels"])
            )
            error = None
        elif config["tokenizer"] == "kmeans":
            codes, books, error = residual_kmeans(
                features, config["levels"], config["codebook_size"], config["seed"]
            )
            np.save(out / "codebooks.npy", books)
        else:
            raise ValueError("tokenizer must be kmeans, rqvae or random")
        dump(
            out / "quantization.json",
            {"reconstruction_mse": error, "method": config["tokenizer"]},
        )
    index = SemanticIndex(codes, config["codebook_size"])
    dump(out / "sid_diagnostics.json", index.diagnostics)
    dump(out / "config.json", config)
    model = make_model(config, mode, index, len(features)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    start, best, logs = 0, -1.0, []
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        # Optimizer state must follow parameters when resuming onto a GPU.
        for values in opt.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        start, best, logs = state["epoch"] + 1, state["best"], state["logs"]
        torch.set_rng_state(state["torch_rng"])
        if device.startswith("cuda") and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    loader = DataLoader(
        examples(train_rows, mode, index, config["max_history"]),
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )
    for epoch in range(start, config["epochs"]):
        model.train()
        loss_sum, tokens = 0.0, 0
        begin = time.perf_counter()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.flatten(), ignore_index=-100
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            count = int(y.ne(-100).sum())
            loss_sum += float(loss.detach()) * count
            tokens += count
        val, _ = predict(
            model, val_rows, mode, index, config, config["k"], config["beam"]
        )
        score = val[f"NDCG@{config['k']}"]
        improved = score > best
        best = max(best, score)
        entry = {
            "epoch": epoch,
            "loss": loss_sum / tokens,
            "validation": val,
            "seconds": time.perf_counter() - begin,
        }
        logs.append(entry)
        print(json.dumps(entry), flush=True)
        checkpoint = {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": opt.state_dict(),
            "config": config,
            "mode": mode,
            "raw_codes": codes.tolist(),
            "epoch": epoch,
            "best": best,
            "logs": logs,
            "data_hash": data_hash,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(checkpoint, out / "best.pt")
        dump(out / "train_log.json", logs)
    dump(
        out / "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": device,
            "data_hash": data_hash,
            "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
        },
    )
    return logs


def load_model(run_dir, device="cpu"):
    state = torch.load(
        Path(run_dir) / "best.pt", map_location="cpu", weights_only=False
    )
    index = SemanticIndex(state["raw_codes"], state["config"]["codebook_size"])
    model = make_model(state["config"], state["mode"], index, len(index.codes)).to(
        device
    )
    model.load_state_dict(state["model"])
    model.eval()
    return model, index, state


def evaluate(data_dir, run_dir, split="test", device="cpu"):
    model, index, state = load_model(run_dir, device)
    if fingerprint(data_dir) != state["data_hash"]:
        raise ValueError("dataset does not match training snapshot")
    cfg = state["config"]
    torch.set_num_threads(cfg.get("threads", 2))
    rows = load_rows(Path(data_dir) / f"{split}.jsonl")
    report, predictions = predict(
        model, rows, state["mode"], index, cfg, cfg["k"], cfg["beam"]
    )
    report["selected_epoch"] = state["epoch"]
    report["protocol"] = (
        "multi-target window, full frozen catalog, no sampled negatives, repeats allowed"
    )
    warm_users = {r["user"] for r in rows if r["history"]}
    for name, subset in [
        ("warm_user", [r for r in predictions if r["user"] in warm_users]),
        ("cold_user", [r for r in predictions if r["user"] not in warm_users]),
        ("cold_target_user", [r for r in predictions if -1 in r["targets"]]),
    ]:
        report[name] = aggregate(subset, cfg["k"]) if subset else None
    dump(Path(run_dir) / f"{split}_metrics.json", report)
    dump(Path(run_dir) / f"{split}_predictions.json", predictions)
    return report


def baselines(data_dir, output, k=12, split="test"):
    n_items = len(json.loads((Path(data_dir) / "items.json").read_text()))
    fitted = ItemBaselines(load_rows(Path(data_dir) / "train.jsonl"), n_items)
    rows = load_rows(Path(data_dir) / f"{split}.jsonl")
    results = {}
    for kind in ("popular", "itemknn"):
        predictions = [
            {**r, "predictions": fitted.predict(r["history"], k, kind)} for r in rows
        ]
        results[kind] = aggregate(predictions, k)
    dump(output, results)
    return results
