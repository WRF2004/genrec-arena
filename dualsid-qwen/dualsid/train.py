from pathlib import Path
import platform
import os
import torch
import transformers
from transformers import Trainer, TrainingArguments, set_seed
from .io import read_json, write_json, fingerprint
from .model import make_tokenizer, create_model, Examples, Collator


def train(data, sid, config, output, resume=None):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("This release supports single-process training; launch with python, not torchrun")
    if sid["data_hash"] != fingerprint(data):
        raise ValueError("SID artifact does not match dataset")
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Run exists; use --resume or choose a new output")
    # Trainer handles checkpoint optimizer, scheduler and RNG restoration.
    signature = fingerprint(dict(data=fingerprint(data), sid=fingerprint(sid), config=config))
    if resume:
        if read_json(output / "manifest.json")["signature"] != signature:
            raise ValueError("Resume config/data/SID mismatch")
    set_seed(config.get("seed", 42))
    tok = make_tokenizer(sid, config["model"], config.get("tiny", False))
    model = create_model(config, tok)
    mode = config.get("mode", "dual")
    train_set = Examples(data["splits"]["train"], sid, tok, config["max_length"], mode)
    val_set = Examples(data["splits"]["val"], sid, tok, config["max_length"], mode, validation=True)
    output.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output), num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 2),
        per_device_eval_batch_size=config.get("batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation", 8),
        learning_rate=config.get("lr", 2e-4), weight_decay=0.01,
        warmup_ratio=0.03, max_grad_norm=1.0, bf16=config.get("bf16", False),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=1, report_to="none", seed=config.get("seed", 42),
        dataloader_num_workers=0, ddp_find_unused_parameters=False,
        save_safetensors=True, use_cpu=config.get("cpu", False))
    trainer = Trainer(model=model, args=args, train_dataset=train_set, eval_dataset=val_set,
                      data_collator=Collator(tok.pad_token_id))
    if trainer.is_world_process_zero():
        tok.save_pretrained(output / "tokenizer")
        write_json(output / "config.json", config)
        write_json(output / "sid.json", sid)
        write_json(output / "manifest.json", dict(signature=signature, data_hash=fingerprint(data),
                   sid_hash=fingerprint(sid), torch=torch.__version__, transformers=transformers.__version__,
                   python=platform.python_version(), train_examples=len(train_set), val_examples=len(val_set)))
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output / "best"))
    trainer.save_state()
    if trainer.is_world_process_zero():
        write_json(output / "history.json", trainer.state.log_history)
    return trainer


def load_run(run, device="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    run = Path(run)
    cfg = read_json(run / "config.json")
    set_seed(cfg.get("seed", 42))
    sid = read_json(run / "sid.json")
    tok = AutoTokenizer.from_pretrained(run / "tokenizer")
    if cfg.get("lora", True):
        # Construct the exact resized base first; new rows come from saved modules.
        base_cfg = {**cfg, "lora": False}
        base = create_model(base_cfg, tok)
        model = PeftModel.from_pretrained(base, run / "best")
    else:
        model = AutoModelForCausalLM.from_pretrained(run / "best")
    model.to(device).eval()
    return model, tok, sid, cfg
