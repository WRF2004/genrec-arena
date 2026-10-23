"""Prompt masks and added-token embeddings are part of the model contract."""
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast


MARKERS = ["<semantic>", "<collaborative>", "<history>", "<item>", "<next>"]


def make_tokenizer(sid, model_name, tiny=False):
    if tiny:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        vocab = {v: i for i, v in enumerate(["<pad>", "<eos>", "<unk>"])}
        tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
                                     pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    else:
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": MARKERS + sid["tokens"]})
    tok.padding_side = "right"
    return tok


def create_model(config, tokenizer):
    if config.get("tiny", False):
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=len(tokenizer), hidden_size=48, intermediate_size=96,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=config["max_length"], tie_word_embeddings=True,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            attention_dropout=0.0))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config["model"], torch_dtype=torch.bfloat16 if config.get("bf16") else torch.float32,
            attn_implementation="sdpa")
        model.resize_token_embeddings(len(tokenizer))
    # Explicitly untie the output projection before PEFT wraps saved modules.
    # Both matrices are then independently trainable and reload without tie ambiguity.
    if model.config.tie_word_embeddings:
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        model.config.tie_word_embeddings = False
    if config.get("lora", True):
        from peft import LoraConfig, get_peft_model
        # Both embedding and output modules MUST train and be saved: new SID tokens
        # start untrained. This costs more memory than a LoRA-only text model.
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=config.get("lora_rank", 16),
            lora_alpha=config.get("lora_rank", 16) * 2, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["embed_tokens", "lm_head"]))
    model.config.use_cache = False
    return model


def path_ids(tokenizer, sid, stream):
    result = {i: tokenizer.convert_tokens_to_ids(path) for i, path in sid["streams"][stream].items()}
    for path in result.values():
        if any(t == tokenizer.unk_token_id for t in path):
            raise ValueError("SID token missing from tokenizer")
    return result


def prompt_ids(tokenizer, paths, history, stream, budget):
    marker = "<semantic>" if stream == "s" else "<collaborative>"
    start = tokenizer.convert_tokens_to_ids([marker, "<history>"])
    end = [tokenizer.convert_tokens_to_ids("<next>")]
    item_token = tokenizer.convert_tokens_to_ids("<item>")
    chunks = [[item_token] + paths[i] for i in history if i in paths]
    while chunks and len(start) + sum(map(len, chunks)) + len(end) > budget:
        chunks.pop(0)
    result = start + [t for chunk in chunks for t in chunk] + end
    if len(result) > budget:
        raise ValueError("max_length too small even for empty prompt")
    return result


class Examples(Dataset):
    def __init__(self, rows, sid, tokenizer, max_length, mode="dual", validation=False):
        self.records = []
        streams = {"dual": ("s", "c"), "semantic": ("s",), "collaborative": ("c",)}[mode]
        for stream in streams:
            paths = path_ids(tokenizer, sid, stream)
            for row in rows:
                targets = row["targets"] if validation else [row["target"]]
                for target in targets:
                    if target not in paths:
                        continue  # loss unavailable; ranking eval keeps OOV in denominator.
                    answer = paths[target] + [tokenizer.eos_token_id]
                    prompt = prompt_ids(tokenizer, paths, row["history"], stream, max_length - len(answer))
                    self.records.append(dict(input_ids=prompt + answer,
                                             labels=[-100] * len(prompt) + answer))
        if not self.records:
            raise ValueError("No in-catalog loss examples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


class Collator:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, rows):
        length = max(len(r["input_ids"]) for r in rows)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for r in rows:
            n = len(r["input_ids"]); p = length - n
            batch["input_ids"].append(r["input_ids"] + [self.pad] * p)
            batch["attention_mask"].append([1] * n + [0] * p)
            batch["labels"].append(r["labels"] + [-100] * p)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
