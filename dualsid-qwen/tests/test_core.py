import csv
import math
import numpy as np
import pytest
import torch
from dualsid.data import demo, prepare
from dualsid.io import fingerprint
from dualsid.sid import addresses, fit, Trie
from dualsid.model import make_tokenizer, create_model, Examples, Collator, path_ids, prompt_ids
from dualsid.retrieval import beam_search, fuse, ranking_metrics


@pytest.fixture
def artifacts(tmp_path):
    data = demo(tmp_path)
    sid = fit(data, levels=2, size=4, dim=8)
    tok = make_tokenizer(sid, "unused", tiny=True)
    return data, sid, tok


def test_temporal_split_and_collaborative_source(tmp_path):
    data = demo(tmp_path)
    assert len(data["splits"]["train"]) == 24 * 7
    assert all(len(s) == 8 for s in data["train_sequences"])
    assert all(len(r["history"]) == 8 for r in data["splits"]["val"])
    assert all(len(r["history"]) == 8 for r in data["splits"]["test"])
    # Test-only item stays OOV and cannot contaminate the frozen catalog.
    with (tmp_path / "events.csv").open("a") as f:
        f.write("u0,new_item,100\nfuture_user,new_item,100\n")
    changed = prepare(tmp_path / "events.csv", tmp_path / "items.csv", tmp_path / "out", 8, 10, 8)
    assert "new_item" not in changed["catalog"]
    assert changed["train_sequences"] == data["train_sequences"]
    assert changed["splits"]["test"][0]["targets"] == ["new_item"]
    assert fit(changed, 2, 4, 8)["streams"] == fit(data, 2, 4, 8)["streams"]


def test_equal_timestamp_does_not_leak(tmp_path):
    items = tmp_path / "items.csv"; events = tmp_path / "events.csv"
    items.write_text("item_id,text\na,alpha\nb,beta\nc,gamma\n")
    events.write_text("user_id,item_id,timestamp\nu,a,1\nu,b,1\nu,c,2\nu,a,3\nu,b,4\nv,b,1\nv,a,2\nv,c,3\nv,b,4\n")
    d = prepare(events, items, tmp_path / "out", 2, 3)
    row = next(r for r in d["splits"]["train"] if r["user"] == "u")
    assert row["history"] == ["a", "b"] and row["target"] == "c"


def test_collision_suffix_handles_more_than_one_digit():
    paths, diag = addresses(np.zeros((17, 2), dtype=int), "s", 2)
    assert len({tuple(p) for p in paths}) == 17
    assert len({len(p) for p in paths}) == 1
    assert diag["suffix_width"] == 5
    assert diag["raw_collision_rate"] == pytest.approx(16 / 17)
    trie = Trie({str(i): p for i, p in enumerate(paths)})
    assert trie.allowed([]) == ["<s0_0>"]
    assert trie.allowed(["invalid"]) == []


def test_reproducible_disjoint_sid(artifacts):
    data, sid, tok = artifacts
    assert sid == fit(data, 2, 4, 8)
    assert set(t for p in sid["streams"]["s"].values() for t in p).isdisjoint(
        t for p in sid["streams"]["c"].values() for t in p)
    for t in sid["tokens"]:
        assert tok.encode(t, add_special_tokens=False) == [tok.convert_tokens_to_ids(t)]


def test_loss_mask_and_trainable_added_tokens(artifacts):
    data, sid, tok = artifacts
    ds = Examples(data["splits"]["train"], sid, tok, 64)
    batch = Collator(tok.pad_token_id)([ds[0], ds[-1]])
    assert (batch["labels"][:, :3] == -100).all()
    cfg = dict(tiny=True, max_length=64, lora=True, lora_rank=4)
    model = create_model(cfg, tok)
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    embedding_grads = [p.grad for n, p in model.named_parameters() if "embed_tokens.modules_to_save" in n]
    head_grads = [p.grad for n, p in model.named_parameters() if "lm_head.modules_to_save" in n]
    assert embedding_grads and all(g is not None and g.abs().sum() > 0 for g in embedding_grads)
    assert head_grads and all(g is not None and g.abs().sum() > 0 for g in head_grads)
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()


def test_beam_returns_only_valid_unique_items(artifacts):
    _, sid, tok = artifacts
    model = create_model(dict(tiny=True, max_length=64, lora=False), tok).eval()
    paths = path_ids(tok, sid, "s")
    prompt = prompt_ids(tok, paths, [], "s", 32)
    result = beam_search(model, prompt, paths, 8)
    assert len(result) == len({i for i, _ in result}) == 8
    assert all(i in sid["catalog"] and math.isfinite(score) for i, score in result)
    assert beam_search(model, prompt, {}, 8) == []


def test_full_width_beam_equals_exhaustive(artifacts):
    _, sid, tok = artifacts
    model = create_model(dict(tiny=True, max_length=64, lora=False), tok).eval()
    paths = dict(list(path_ids(tok, sid, "s").items())[:5])
    prompt = prompt_ids(tok, paths, [], "s", 32)
    actual = beam_search(model, prompt, paths, 100)
    expected = []
    with torch.inference_mode():
        for item, path in paths.items():
            x = torch.tensor([prompt + path])
            lp = model(x).logits.float().log_softmax(-1)
            score = sum(float(lp[0, len(prompt) + j - 1, t]) for j, t in enumerate(path))
            expected.append((item, score))
    assert [i for i, _ in actual] == [i for i, _ in sorted(expected, key=lambda p: (-p[1], p[0]))]
    assert dict(actual) == pytest.approx(dict(expected), abs=1e-5)


def test_metrics_oov_and_rrf():
    m = ranking_metrics([["a", "a", "b"]], [["b", "unknown"]], ["a", "b"], 2)
    assert m["recall@2"] == .5
    assert m["oov_target_rate"] == .5
    assert m["ndcg@2"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert fuse([("a", 0)], [("b", 0)], alpha=1) == ["a"]
    assert fuse([("a", 0)], [("b", 0)], alpha=0) == ["b"]
    with pytest.raises(ValueError):
        fuse([], [], -1)


def test_embedding_validation(artifacts):
    data, _, _ = artifacts
    with pytest.raises(ValueError):
        fit(data, embeddings=np.ones((2, 2)))
    assert fingerprint(data) == fingerprint(dict(reversed(list(data.items()))))
