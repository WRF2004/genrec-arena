"""Frozen Qwen content encoding; a baseline pooling recipe, not a tuned embedder."""
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@torch.inference_mode()
def encode(data, output, model_name="Qwen/Qwen2.5-1.5B", device="cuda", batch_size=8, max_length=128):
    if batch_size < 1 or max_length < 1:
        raise ValueError("Positive batch size and max length required")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
    model.to(device).eval()
    result = []
    for start in range(0, len(data["texts"]), batch_size):
        batch = tok(data["texts"][start:start + batch_size], padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt").to(device)
        h = model(**batch).last_hidden_state.float()
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
        result.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, item_ids=np.array(data["catalog"]), embeddings=np.concatenate(result))
