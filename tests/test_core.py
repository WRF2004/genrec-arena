import csv
import json
import math

import numpy as np
import pytest
import torch

from genrec.data import convert, load_rows, prepare, synthetic
from genrec.engine import collate, evaluate, examples, load_model, train
from genrec.metrics import metrics
from genrec.models import CausalTransformer
from genrec.tokenizer import SemanticIndex, fit_rqvae, residual_kmeans


def test_metrics_cold_duplicates():
    result = metrics([2, 2, 3, 4], [2, 3, -1, -1], 4)
    assert result["Recall@4"] == 0.5
    assert result["MAP@4"] == pytest.approx((1 + 2 / 3) / 4)
    assert metrics([3, 2], [2, 3], 2)["NDCG@2"] == 1
    assert metrics([], [-1], 12)["MAP@12"] == 0


def test_collision_and_beam():
    torch.set_num_threads(1)
    index = SemanticIndex(np.zeros((19, 2), dtype=int), 2)
    assert len(index.leaves) == 19
    assert index.diagnostics["suffix_digits"] == 5
    model = CausalTransformer(index.vocab, 100, dim=16, heads=2, layers=1, dropout=0)
    assert set(index.beam_search(model, [], k=19, beam=32)) == set(range(19))
    with pytest.raises(ValueError):
        index.beam_search(model, [], k=12, beam=4)


def test_causal_and_padding():
    torch.manual_seed(42)
    model = CausalTransformer(30, 20, dim=16, heads=2, layers=1, dropout=0).eval()
    a = model(torch.tensor([[1, 4, 5, 6]]))
    b = model(torch.tensor([[1, 4, 9, 8]]))
    assert torch.allclose(a[:, :2], b[:, :2], atol=1e-6)
    padded = model(torch.tensor([[1, 4, 5, 6, 0, 0]]))
    assert torch.allclose(a, padded[:, :4], atol=1e-6)


def test_target_loss_mask():
    index = SemanticIndex([[0, 1], [1, 0]], 2)
    x, y = collate(examples([{"history": [0], "targets": [1]}], "generative", index, 4))
    assert (y != -100).sum() == index.depth
    assert y[0, -index.depth :].tolist() == index.codes[1].tolist()
    assert x[0, -index.depth].item() == index.SEP


def test_time_split_and_cold_catalog(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text(
        "user_id,item_id,timestamp,category\nu,a,1,A\nu,b,2,B\nu,a,3,A\nu,c,4,C\nu,d,6,D\nv,e,6,E\n"
    )
    out = tmp_path / "prepared"
    report = prepare(path, out, 4, 6)
    assert json.loads((out / "items.json").read_text()) == ["a", "b"]
    assert load_rows(out / "val.jsonl")[0]["targets"] == [-1]
    assert load_rows(out / "test.jsonl")[1]["history"] == []
    assert report["cold_target_fraction"]["test"] == 1


def test_hm_ids(tmp_path):
    raw = tmp_path / "hm.csv"
    raw.write_text("customer_id,article_id,t_dat\nu,001,2020-09-01\nu,002,2020-09-01\n")
    dest = tmp_path / "converted.csv"
    convert(raw, dest, "hm")
    rows = list(csv.DictReader(dest.open()))
    assert rows[0]["item_id"] == "001"
    assert rows[0]["timestamp"] == rows[1]["timestamp"]


def test_quantizers():
    torch.set_num_threads(1)
    x = np.random.default_rng(1).normal(size=(16, 8)).astype(np.float32)
    codes, _, error = residual_kmeans(x, 2, 4)
    assert codes.shape == (16, 2) and math.isfinite(error)
    rcodes, state, recon = fit_rqvae(x, 2, 4, epochs=2)
    assert rcodes.shape == (16, 2) and math.isfinite(recon)
    assert all(torch.isfinite(v).all() for v in state.values())


def test_training_resume_and_evaluation(tmp_path):
    raw, data = tmp_path / "raw.csv", tmp_path / "data"
    synthetic(raw, users=8, items=24)
    prepare(raw, data, 16000, 20000, max_history=3)
    cfg = {
        "seed": 42,
        "epochs": 1,
        "batch_size": 32,
        "lr": 0.002,
        "dim": 16,
        "heads": 2,
        "layers": 1,
        "dropout": 0.1,
        "max_history": 3,
        "levels": 2,
        "codebook_size": 4,
        "tokenizer": "kmeans",
        "k": 5,
        "beam": 8,
        "threads": 1,
    }
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps(cfg))
    run = tmp_path / "run"
    train(data, run, config)
    assert 0 <= evaluate(data, run)["NDCG@5"] <= 1
    cfg["epochs"] = 2
    config.write_text(json.dumps(cfg))
    train(data, run, config, resume=True)
    train(data, tmp_path / "continuous", config)
    a = torch.load(run / "last.pt", weights_only=False)
    b = torch.load(tmp_path / "continuous" / "last.pt", weights_only=False)
    assert all(torch.equal(v, b["model"][k]) for k, v in a["model"].items())
    model, index, _ = load_model(run)
    assert len(index.beam_search(model, [], 5, 8, 3)) == 5
    cfg["seed"] = 99
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="mismatch"):
        train(data, run, config, resume=True)
