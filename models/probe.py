import argparse
import json
import yaml
import numpy as np
import torch
import os
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from utils.seeding import set_global_seed, log_run_meta


def pick_dtype(precision: str, device: torch.device):
    precision = (precision or "").lower()
    if device.type == "cuda":
        if precision in ("bf16", "bfloat16") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if precision in ("fp16", "float16", "half"):
            return torch.float16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def ensure_padding(tok):
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "right"


@torch.inference_mode()
def pooled_hidden_batch(model, tok, texts, layer, batch_size, max_length, pool="mean"):
    device = model.device
    outs = []
    i = 0
    while i < len(texts):
        bs = min(batch_size, len(texts) - i)
        chunk = texts[i:i+bs]
        try:
            enc = tok(chunk, return_tensors="pt", truncation=True,
                      padding=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer]  # [B, T, H]
            if pool == "cls":
                pooled = hs[:, 0, :]
            else:
                mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
                summed = (hs * mask).sum(dim=1)             # [B, H]
                counts = mask.sum(dim=1).clamp(min=1)       # [B, 1]
                pooled = summed / counts
            outs.append(pooled.float().cpu().numpy())
            i += bs
        except torch.cuda.OutOfMemoryError:
            # reduce batch size and retry
            torch.cuda.empty_cache()
            if bs == 1:
                raise  # even single-item fails → let caller handle (CPU fallback)
            batch_size = max(1, bs // 2)
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, model.config.hidden_size), dtype=np.float32)


def load_texts(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            j = json.loads(ln)
            t = j.get("text", "")
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
    return texts


def main(args):
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_global_seed(int(cfg.get("seed", 9172)))
    log_run_meta(out_dir="probes", cfg=cfg, extras={"script": "probe"})

    # performance knobs / defaults
    user_precision = cfg.get("precision", "")
    user_attn_impl = getattr(cfg, "attn_impl", None) or cfg.get("attn_impl", None)  # optional
    batch = int(getattr(args, "batch", 0) or 8)          # safe default for 4GB
    max_len = int(getattr(args, "max_len", 0) or 128)    # 128 tokens is much lighter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = pick_dtype(user_precision, device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        # allow SDPA to pick memory-efficient kernels
        try:
            torch.nn.attention.sdpa_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=True)
        except Exception:
            pass
        # optional env hint for fragmentation
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # --- Separate model id vs tokenizer id ---
    model_id = cfg["model_name"]  # HF base model or PEFT adapter dir
    tok_id = cfg.get("tokenizer_name", cfg["model_name"])

    model_kwargs = {"dtype": dtype}
    if user_attn_impl in ("sdpa", "eager"):  # optional override
        model_kwargs["attn_implementation"] = user_attn_impl

    # Load tokenizer (can be local HF tokenizer folder)
    tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    ensure_padding(tok)

    # Try to load model_id as a full HF model first
    try:
        model = AutoModel.from_pretrained(model_id, **model_kwargs)
        model = model.to(device).eval()
    except Exception:
        # If that fails, assume model_id is a PEFT adapter directory.
        # In that case, load base model from tok_id and attach the adapter.
        base_model = AutoModel.from_pretrained(tok_id, **model_kwargs)
        base_model = base_model.to(device).eval()

        try:
            _ = PeftConfig.from_pretrained(model_id)
        except Exception as e:
            raise ValueError(
                f"`model_name`='{model_id}' is neither a valid HF model nor a PEFT adapter dir. "
                f"Original error: {type(e).__name__}: {e}"
            )

        model = PeftModel.from_pretrained(base_model, model_id)
        model = model.to(device).eval()

    H = load_texts(cfg["data"]["anchors"])
    N = load_texts(cfg["data"]["neutrals"])

    # determine valid layer index
    tmp = tok("hi", return_tensors="pt").to(device)
    with torch.inference_mode():
        tmp_out = model(**tmp, output_hidden_states=True)
    Lp1 = len(tmp_out.hidden_states)
    layer_cfg = int(cfg["drift"]["layer"])
    layer = layer_cfg if layer_cfg >= 0 else (Lp1 + layer_cfg)
    assert 0 <= layer < Lp1, f"Invalid layer {layer_cfg}; hidden_states has length {Lp1}"

    # Try GPU, back off batch on OOM, then CPU fallback if necessary
    try:
        XH = pooled_hidden_batch(model, tok, H, layer, batch, max_len, pool="mean")
        XN = pooled_hidden_batch(model, tok, N, layer, batch, max_len, pool="mean")
    except torch.cuda.OutOfMemoryError:
        print("[warn] GPU OOM even at tiny batch — falling back to CPU.")
        torch.cuda.empty_cache()
        device = torch.device("cpu")
        model = model.to(device)
        XH = pooled_hidden_batch(model, tok, H, layer, batch_size=8,
                                 max_length=max_len, pool="mean")
        XN = pooled_hidden_batch(model, tok, N, layer, batch_size=8,
                                 max_length=max_len, pool="mean")

    X = np.vstack([XH, XN]).astype(np.float32)
    y = np.concatenate([np.ones(len(H), dtype=np.int32), np.zeros(len(N), dtype=np.int32)])

    clf = LogisticRegression(max_iter=500, solver="lbfgs").fit(X, y)
    v = clf.coef_.astype(np.float32)[0]
    v /= (np.linalg.norm(v) + 1e-12)

    save_dir = os.path.dirname(args.save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    np.save(args.save, v)
    print(
        f"Saved concept vector to {args.save} (dim={v.shape[0]}, device={model.device}, dtype=float32, layer={layer}, batch={batch}, max_len={max_len})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--save", required=True)
    p.add_argument("--batch", type=int, default=8,
                   help="encode batch size (defaults to 8 for 4GB GPUs)")
    p.add_argument("--max_len", type=int, default=128, help="encode max_length (defaults to 128)")
    main(p.parse_args())
