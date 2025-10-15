# eval/eval_perplexity.py
import argparse
import glob
import math
import os
import json
import yaml
import torch
import numpy as np
from typing import Iterable, List
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def _ensure_pad(tok):
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "right"


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                j = json.loads(ln)
                yield j
            except Exception:
                # treat raw line as {"text": line}
                yield {"text": ln.strip()}


def _load_texts_local(spec: str) -> List[str]:
    # file
    if os.path.isfile(spec):
        return [ex.get("text") or ex.get("prompt") or ex.get("instruction") or ex.get("question")
                for ex in _iter_jsonl(spec)]
    # dir
    if os.path.isdir(spec):
        preferred = os.path.join(spec, "u_dev.jsonl")
        files = [preferred] if os.path.isfile(preferred) else sorted(
            glob.glob(os.path.join(spec, "*.jsonl")))
        out = []
        for fp in files:
            out.extend([ex.get("text") or ex.get("prompt") or ex.get("instruction") or ex.get("question")
                        for ex in _iter_jsonl(fp)])
        return out
    # glob
    if any(ch in spec for ch in "*?[]"):
        out = []
        for fp in sorted(glob.glob(spec)):
            out.extend([ex.get("text") or ex.get("prompt") or ex.get("instruction") or ex.get("question")
                        for ex in _iter_jsonl(fp)])
        return out
    return []


def _load_u_dev(spec: str, sample_n: int, seed: int) -> List[str]:
    # try local first
    local = _load_texts_local(spec)
    local = [t for t in local if isinstance(t, str) and t.strip()]
    if local:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(local))[: min(sample_n, len(local))]
        return [local[i] for i in idx]

    # HF dataset id; try as HF repo, else "json" builder if looks like file/glob
    try:
        ds = load_dataset(spec, split="train")
    except Exception:
        if spec.lower().endswith((".jsonl", ".json")):
            ds = load_dataset("json", data_files=spec, split="train")
        else:
            raise
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(min(sample_n, len(ds))))
    # Prefer common text fields
    cand_keys = ["text", "prompt", "instruction", "question", "content", "inputs", "input"]

    def get_text(ex):
        for k in cand_keys:
            v = ex.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return None
    return [t for t in (get_text(x) for x in ds) if isinstance(t, str) and t.strip()]


@torch.no_grad()
def perplexity(model, tok, texts: List[str], batch_size: int = 2, max_length: int = 128) -> float:
    _ensure_pad(tok)
    device = model.device
    total_nll = 0.0
    total_tokens = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        # labels: ignore pad tokens to avoid counting them
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        out = model(**enc, labels=labels)
        # out.loss is mean CE over non -100; recover token sum:
        # nll_batch = loss * (# of non-ignored tokens)
        non_ignored = (labels != -100).sum().item()
        total_nll += out.loss.item() * max(non_ignored, 1)
        total_tokens += non_ignored
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def main(args):
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    model_name = cfg["model_name"]
    # load data (HF id OR local file/dir/glob)
    u_spec = cfg["data"]["unlabeled_stream"]
    sample_n = int(cfg["eval"]["u_dev_sample"])
    u_dev_texts = _load_u_dev(u_spec, sample_n=sample_n, seed=9172)
    if not u_dev_texts:
        raise FileNotFoundError(f"No texts found for spec: {u_spec}")

    # model/tokenizer + minor safety knobs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    _ensure_pad(tok)

    ppl = perplexity(model, tok, u_dev_texts, batch_size=8, max_length=512)
    print(f"Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args())
