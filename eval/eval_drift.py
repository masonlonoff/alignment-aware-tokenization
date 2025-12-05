# eval/eval_drift.py
"""
Drift evaluator (full):
- Computes per-text concept logits for Hazard (H) and Neutral (N).
- Reports μ/σ² for each, and Drift Gap = (μ_H - μ_N) ↑.
- Saves arrays for histogram plots.

Outputs (stdout):
  Neutral drift mean: <float>
  Neutral drift var:  <float>
  Hazard  drift mean: <float>
  Hazard  drift var:  <float>
  Drift gap (H-N):    <float>

Also writes:
  <out_dir>/drift_neutral.npy
  <out_dir>/drift_hazard.npy
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import torch
import yaml
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModel


def _read_texts(path: str, key: str = "text") -> List[str]:
    """Read newline-delimited JSON and return non-empty `key` fields."""
    xs: List[str] = []
    if not os.path.exists(path):
        return xs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            t = (j.get(key) or "").strip()
            if t:
                xs.append(t)
    return xs


def _load_with_dtype(model_id: str, dtype: torch.dtype):
    """
    Helper to load a model using `dtype` (newer HF) with a
    fallback to `torch_dtype` (older HF) for compatibility.
    """
    try:
        return AutoModel.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        # Older transformers versions don't know `dtype=`
        return AutoModel.from_pretrained(model_id, torch_dtype=dtype)


def _load_model_and_tokenizer(cfg):
    """
    Load tokenizer from `tokenizer_name` (can be a local tokenizer folder),
    and load model from `model_name`. If `model_name` is a LoRA/PEFT adapter
    directory, we load the base model from `tokenizer_name` and then apply the adapter.

    This version is device-safe: it runs on GPU if available, otherwise CPU.
    """
    model_id = cfg["model_name"]  # HF id or PEFT adapter dir
    tok_id = cfg.get("tokenizer_name", cfg["model_name"])

    # Decide device & dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Tokenizer may live in a local folder -> always safe to load via AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)

    # First, try to load `model_id` as a full HF model
    try:
        model = _load_with_dtype(model_id, dtype=dtype)
        model.to(device).eval()
        return model, tok
    except Exception:
        # If that fails, assume `model_id` is a PEFT adapter directory.
        # Load base model from `tok_id` (which in your configs points to a real base model id)
        base_model = _load_with_dtype(tok_id, dtype=dtype)
        base_model.to(device).eval()

        try:
            _ = PeftConfig.from_pretrained(model_id)
        except Exception as e:
            raise ValueError(
                f"`model_name`='{model_id}' is neither a valid HF model nor a PEFT adapter dir. "
                f"Original error: {type(e).__name__}: {e}"
            )

        model = PeftModel.from_pretrained(base_model, model_id)
        model.to(device).eval()
        # Optional: merge LoRA for speed (commented out by default)
        # model = model.merge_and_unload()
        return model, tok


@torch.no_grad()
def drift_score(
    model: torch.nn.Module,
    tok,
    texts: List[str],
    v_np: np.ndarray,
    layer: int,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Compute cosine-similarity-based concept scores for each text.

    Args:
        model: HF model with `output_hidden_states=True` support.
        tok:   Corresponding tokenizer.
        texts: List of input strings.
        v_np:  Concept direction as a 1D numpy array.
        layer: Which hidden-state layer index to probe.
        batch_size: Number of texts per forward pass.

    Returns:
        np.ndarray of shape [len(texts)] with drift scores.
    """
    if not texts:
        return np.empty((0,), dtype=np.float32)

    device = next(model.parameters()).device

    # Normalize concept vector on the model device
    v = torch.as_tensor(v_np, dtype=torch.float32, device=device).view(-1)
    v = v / (v.norm() + 1e-9)

    scores: List[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(device)

        out = model(**enc, output_hidden_states=True)
        # hidden_states[layer]: [B, T, H]
        h = out.hidden_states[layer]
        # mean-pool across tokens -> [B, H]
        h_mean = h.mean(dim=1).to(torch.float32)

        # cosine similarity with v per example
        sim = torch.nn.functional.cosine_similarity(
            h_mean, v.unsqueeze(0).expand_as(h_mean), dim=-1
        )
        scores.extend(sim.tolist())

    return np.asarray(scores, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--probe", required=True, help="Concept vector .npy")
    ap.add_argument("--out_dir", default="runs/latest")
    ap.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size used when encoding texts for drift scoring.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model, tok = _load_model_and_tokenizer(cfg)

    # Load and normalize concept vector
    v = np.load(args.probe, allow_pickle=True).reshape(-1)
    v = v / (np.linalg.norm(v) + 1e-9)

    # using anchors as hazard texts (behavior descriptions)
    N = _read_texts(cfg["data"]["neutrals"])
    H = _read_texts(cfg["data"]["anchors"])

    layer = cfg["drift"]["layer"]
    sN = drift_score(model, tok, N, v, layer, batch_size=args.batch_size)
    sH = drift_score(model, tok, H, v, layer, batch_size=args.batch_size)

    np.save(os.path.join(args.out_dir, "drift_neutral.npy"), sN)
    np.save(os.path.join(args.out_dir, "drift_hazard.npy"), sH)

    muN = float(np.mean(sN)) if len(sN) else 0.0
    varN = float(np.var(sN)) if len(sN) else 0.0
    muH = float(np.mean(sH)) if len(sH) else 0.0
    varH = float(np.var(sH)) if len(sH) else 0.0
    gap = muH - muN

    print(f"Neutral drift mean: {muN:.6f}")
    print(f"Neutral drift var:  {varN:.6f}")
    print(f"Hazard  drift mean: {muH:.6f}")
    print(f"Hazard  drift var:  {varH:.6f}")
    print(f"Drift gap (H-N):    {gap:.6f}")


if __name__ == "__main__":
    main()
