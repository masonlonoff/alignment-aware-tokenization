# eval/eval_perplexity.py
import argparse
import glob
import json
import math
import os
from typing import Iterable, List, Optional

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.seeding import set_global_seed, log_run_meta, tokenizer_hash


def _ensure_pad(tok):
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "right"  # right padding is fine for PPL


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                yield json.loads(ln)
            except Exception:
                yield {"text": ln.strip()}


def _extract_text(j: dict) -> Optional[str]:
    cand_keys = ["text", "prompt", "instruction", "question", "content", "inputs", "input"]
    for k in cand_keys:
        v = j.get(k)
        if isinstance(v, str) and v.strip():
            return v
    for v in j.values():
        if isinstance(v, str) and v.strip():
            return v
    return None


def _load_texts_local(spec: str) -> List[str]:
    if os.path.isfile(spec):
        return [t for t in (_extract_text(ex) for ex in _iter_jsonl(spec)) if t]
    if os.path.isdir(spec):
        preferred = os.path.join(spec, "u_dev.jsonl")
        files = [preferred] if os.path.isfile(preferred) else sorted(
            glob.glob(os.path.join(spec, "*.jsonl"))
        )
        out: List[str] = []
        for fp in files:
            out.extend([t for t in (_extract_text(ex) for ex in _iter_jsonl(fp)) if t])
        return out
    if any(ch in spec for ch in "*?[]"):
        out: List[str] = []
        for fp in sorted(glob.glob(spec)):
            out.extend([t for t in (_extract_text(ex) for ex in _iter_jsonl(fp)) if t])
        return out
    return []


def _sample_texts(texts: List[str], sample_n: int, seed: int) -> List[str]:
    if not texts or sample_n is None or sample_n <= 0 or sample_n >= len(texts):
        return texts
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(texts))[:sample_n]
    return [texts[i] for i in idx]


@torch.no_grad()
def perplexity(model, tok, texts: List[str], batch_size: int = 8, max_length: int = 256) -> float:
    _ensure_pad(tok)
    device = model.device
    total_nll = 0.0
    total_tokens = 0
    try:
        model.config.use_cache = False  # more stable loss with full context
    except Exception:
        pass

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        out = model(**enc, labels=labels)
        non_ignored = int((labels != -100).sum().item())
        total_nll += float(out.loss.item()) * max(non_ignored, 1)
        total_tokens += non_ignored

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def _pick_attn_impl() -> str:
    # Avoid FlashAttn2 requirement
    return "sdpa" if torch.cuda.is_available() else "eager"


def _pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def main(args):
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    model_name = cfg.get("model_name")
    tokenizer_name = cfg.get("tokenizer_name", model_name)
    data_spec = cfg["data"]["unlabeled_stream"]

    seed = int(cfg.get("seed", 9172))
    bs = int(args.batch_size or cfg.get("eval", {}).get("batch_size", 8))
    max_len = int(args.max_length or cfg.get("eval", {}).get("max_length", 256))
    sample_n = int(args.sample_n or cfg.get("eval", {}).get("u_dev_sample", 1024))

    # seeding
    set_global_seed(seed)

    # data
    texts = _load_texts_local(data_spec)
    if not texts:
        raise FileNotFoundError(f"No texts found for spec: {data_spec}")
    texts = _sample_texts(texts, sample_n=sample_n, seed=seed)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn_impl = _pick_attn_impl()
    dtype = _pick_dtype()

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    _ensure_pad(tok)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation=attn_impl,   # prevents FlashAttn2 import error
        dtype=dtype,                     # use `dtype`, not deprecated `torch_dtype`
        low_cpu_mem_usage=True,
    ).to(device).eval()

    ppl = perplexity(model, tok, texts, batch_size=bs, max_length=max_len)
    print(f"Perplexity: {ppl:.3f}")

    # from tokenizers.bpe_search import tokens_per_char
    # tpc = tokens_per_char(tok, texts)
    # print(f"TPC: {tpc:.5f}")

    out_dir = cfg.get("out_dir", None)
    if out_dir:
        extras = {
            "num_texts": len(texts),
            "batch_size": bs,
            "max_length": max_len,
            "attn_impl": attn_impl,
            "dtype": str(dtype).replace("torch.", ""),
            "perplexity": ppl,
            "tokenizer_hash": tokenizer_hash(tokenizer_name),
        }
        log_run_meta(out_dir, cfg, extras=extras)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--sample_n", type=int, default=1024)
    main(p.parse_args())
