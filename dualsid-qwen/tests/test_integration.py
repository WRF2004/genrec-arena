from pathlib import Path
import torch
import pytest
from dualsid.data import demo
from dualsid.sid import fit
from dualsid.model import Examples, Collator
from dualsid.train import train, load_run
from dualsid.retrieval import evaluate


def test_train_save_reload_resume_and_evaluate(tmp_path):
    data = demo(tmp_path / "data")
    sid = fit(data, 2, 4, 8)
    cfg = dict(model="Qwen/Qwen2.5-1.5B", tiny=True, lora=True, lora_rank=4,
               max_length=64, epochs=2, batch_size=32, gradient_accumulation=1,
               lr=.003, bf16=False, cpu=True, seed=42)
    run = tmp_path / "run"
    trainer = train(data, sid, cfg, run)
    model, tok, saved_sid, saved_cfg = load_run(run)
    batch = Collator(tok.pad_token_id)([Examples(data["splits"]["train"], sid, tok, 64)[0]])
    trainer.model.eval()
    with torch.inference_mode():
        expected = trainer.model(**batch).logits
        actual = model(**batch).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    checkpoint = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))[0]
    resumed = train(data, sid, cfg, run, str(checkpoint))
    assert resumed.state.global_step == trainer.state.global_step
    resumed.model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(resumed.model(**batch).logits, expected, rtol=0, atol=0)
    result = evaluate(model, tok, saved_sid, data, "test", saved_cfg, k=3, beam=4)
    assert result["metrics"]["users"] == 24
    assert all(len(r["items"]) == 3 for r in result["predictions"])
    with pytest.raises(ValueError, match="Resume config"):
        train(data, sid, {**cfg, "seed": 43}, run, str(checkpoint))
    with pytest.raises(ValueError, match="Run exists"):
        train(data, sid, cfg, run)


def test_frozen_encoder_local_qwen_snapshot(tmp_path):
    import numpy as np
    from dualsid.encode import encode
    from dualsid.model import make_tokenizer, create_model
    data = demo(tmp_path / "data")
    sid = fit(data, 2, 4, 8)
    tok = make_tokenizer(sid, "unused", tiny=True)
    base = create_model(dict(tiny=True, max_length=64, lora=False), tok)
    local = tmp_path / "base"
    base.save_pretrained(local)
    tok.save_pretrained(local)
    output = tmp_path / "content.npz"
    encode(data, output, str(local), device="cpu", batch_size=8)
    z = np.load(output, allow_pickle=False)
    assert z["item_ids"].tolist() == data["catalog"]
    assert z["embeddings"].shape == (32, 48)
    np.testing.assert_allclose(np.linalg.norm(z["embeddings"], axis=1), 1, atol=1e-6)
    assert fit(data, 2, 4, 8, embeddings=z["embeddings"])["config"]["content"].startswith("external")
