# models/lora_drift.py
import argparse
import json
import yaml
import numpy as np
import torch
import os
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from utils.seeding import set_global_seed, log_run_meta

# ---------------------------
# Helpers
# ---------------------------


def _pick_dtype_and_device(precision: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prec = (precision or "").lower()
    if device.type == "cuda":
        if prec in ("bf16", "bfloat16") and torch.cuda.is_bf16_supported():
            return torch.bfloat16, device
        if prec in ("fp16", "float16", "half"):
            return torch.float16, device
        return (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16), device
    return torch.float32, device


def _ensure_padding(tok):
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "right"


def _safe_load_jsonl_texts(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                j = json.loads(ln)
                t = j.get("text", "")
            except Exception:
                t = ln
            if isinstance(t, str) and t.strip():
                rows.append(t.strip())
    return rows


def _leaf_attr_names(model):
    names = set()
    for n, _ in model.named_modules():
        names.add(n.split(".")[-1])
    return names


def _guess_targets(model_name: str, model):
    present = _leaf_attr_names(model)
    name = (model_name or "").lower()
    mt = str(getattr(getattr(model, "config", None), "model_type", "")).lower()

    rules = [
        (("llama", "mistral", "qwen", "qwen2", "gemma", "phi", "phi-3"),
         ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        (("opt",),
         ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]),
        (("falcon", "gpt-neox", "neox", "pythia"),
         ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]),
        (("gpt2", "gpt-j", "gpt-neo"),
         ["c_attn", "c_proj", "mlp.c_fc", "mlp.c_proj"]),
    ]
    chosen = []
    for keys, cand in rules:
        if any(k in mt for k in keys) or any(k in name for k in keys):
            chosen = cand
            break
    if not chosen:
        chosen = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                  "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",
                  "c_attn", "c_proj", "out_proj", "fc1", "fc2", "mlp.c_fc", "mlp.c_proj"]
    targets = [t for t in chosen if any(t in n for n in present)]
    if not targets:
        targets = ["query_key_value", "dense", "dense_h_to_4h",
                   "dense_4h_to_h", "q_proj", "k_proj", "v_proj", "o_proj"]
    return sorted(set(targets))

# ---------------------------
# Streaming unlabeled text (LOCAL ONLY)
# ---------------------------


def _iter_jsonl_file(path, max_items):
    cnt = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if cnt >= max_items:
                break
            if not ln.strip():
                continue
            try:
                ex = json.loads(ln)
                t = ex.get("text") or ex.get("prompt") or ex.get(
                    "instruction") or ex.get("question")
            except Exception:
                t = ln
            if isinstance(t, str) and t.strip():
                yield t.strip()
                cnt += 1


def stream_text_local(spec, max_items=20000):
    """
    Stream texts from local JSONL sources ONLY.

    Accepts:
      - A local file path:           '/path/to/u_train.jsonl'
      - A local directory path:      '/path/to/unlabeled/'  (auto-uses u_train.jsonl, u_dev.jsonl, then *.jsonl)
      - A glob pattern:              '/path/to/unlabeled/*.jsonl'

    Notes:
      - No Hugging Face dataset IDs are supported here.
      - The function yields up to `max_items` texts per file in iteration order.
    """
    # Glob pattern
    if any(ch in spec for ch in ["*", "?", "["]):
        files = sorted(glob.glob(spec))
        for fp in files:
            yield from _iter_jsonl_file(fp, max_items)
        return

    # Single file
    if os.path.isfile(spec):
        yield from _iter_jsonl_file(spec, max_items)
        return

    # Directory
    if os.path.isdir(spec):
        candidates = []
        for name in ["u_train.jsonl", "u_dev.jsonl"]:
            cand = os.path.join(spec, name)
            if os.path.isfile(cand):
                candidates.append(cand)
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(spec, "*.jsonl")))
        for fp in candidates:
            yield from _iter_jsonl_file(fp, max_items)
        return

    # Not found
    raise FileNotFoundError(f"Local path not found (file/dir/glob expected): {spec}")

# ---------------------------
# Drift penalty
# ---------------------------


@torch.no_grad()
def drift_penalty(model, tok, v_vec, texts, layer, margin):
    v = torch.as_tensor(v_vec, dtype=model.dtype, device=model.device).view(-1)
    scores = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=256, padding=False)
        enc = {k: v_.to(model.device) for k, v_ in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[layer][0]    # [T, H]
        h_mean = hs.mean(dim=0)             # [H]
        s = torch.dot(h_mean, v)
        scores.append(torch.relu(s - margin) ** 2)
    return torch.stack(scores).mean() if scores else torch.tensor(0.0, device=model.device, dtype=model.dtype)


def _load_probe_vector():
    for p in ["probes/v_layer.npy", "probes/v_layer.pt.npy", "probes/v_layer.pt"]:
        if os.path.exists(p):
            v = np.load(p, allow_pickle=True)
            return np.asarray(v).squeeze()
    raise FileNotFoundError(
        "Probe vector not found in probes/ (looked for v_layer.npy / .pt.npy / .pt)")

# ---------------------------
# Main
# ---------------------------


def main(args):
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_global_seed(int(cfg.get("seed", 9172)))
    log_run_meta(out_dir=args.save, cfg=cfg, extras={"script": "lora_drift"})

    dtype, device = _pick_dtype_and_device(cfg.get("precision", ""))
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    _ensure_padding(tok)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], dtype=dtype).to(device).train()

    cfg_targets = cfg.get("lora", {}).get("target_modules")
    targets = cfg_targets if cfg_targets else _guess_targets(cfg["model_name"], model)
    lcfg = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        target_modules=targets,
    )
    try:
        model = get_peft_model(model, lcfg)
    except ValueError as e:
        raise ValueError(
            f"LoRA target_modules not found in base model. Tried {targets}. "
            f"Set lora.target_modules in your YAML for {cfg['model_name']}."
        ) from e

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))
    sch = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=int(cfg["train"]["warmup_steps"]),
        num_training_steps=int(cfg["train"]["steps"]),
    )

    neutrals = _safe_load_jsonl_texts(cfg["data"]["neutrals"])
    v = _load_probe_vector()
    # v = v / (np.linalg.norm(v) + 1e-9)
    stream = stream_text_local(
        cfg["data"]["unlabeled_stream"],
        max_items=int(cfg["eval"]["u_dev_sample"])
    )

    steps = int(cfg["train"]["steps"])
    lam = float(cfg["drift"]["lambda"])
    margin = float(cfg["drift"]["margin"])
    layer = int(cfg["drift"]["layer"])
    neutrals_slice = neutrals[: max(1, min(64, len(neutrals)))]

    model.train()
    for step, text in enumerate(stream):
        if step >= steps:
            break
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        enc = {k: v_.to(device) for k, v_ in enc.items()}
        out = model(**enc, labels=enc["input_ids"])
        lm_loss = out.loss

        drift = drift_penalty(model, tok, v, neutrals_slice, layer, margin)
        loss = lm_loss + lam * drift

        loss.backward()
        opt.step()
        sch.step()
        model.zero_grad(set_to_none=True)

        if step % 50 == 0:
            print(f"step {step} | lm {lm_loss.item():.3f} | drift {drift.item():.3f} | targets={targets}")

    os.makedirs(args.save, exist_ok=True)
    model.save_pretrained(args.save)
    tok.save_pretrained(args.save)
    print("Saved LoRA adapter to", args.save)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--save", required=True)
    main(p.parse_args())
