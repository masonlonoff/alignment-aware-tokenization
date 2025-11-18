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
import numpy as np
import yaml
import torch
from typing import List
from transformers import AutoTokenizer, AutoModel


def _read_texts(path: str, key="text") -> List[str]:
    xs = []
    if not os.path.exists(path):
        return xs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            t = (j.get(key) or "").strip()
            if t:
                xs.append(t)
    return xs


@torch.no_grad()
def drift_score(model, tok, texts, v: np.ndarray, layer: int) -> np.ndarray:
    # make sure v is a flat float32 tensor on the right device
    v_t = torch.as_tensor(v, dtype=torch.float32, device=model.device).reshape(-1)
    scores = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        out = model(**ids, output_hidden_states=True)
        h = out.hidden_states[layer][0]              # [T, H]
        h_mean = h.mean(dim=0).to(torch.float32)     # cast to float32
        scores.append(torch.dot(h_mean, v_t).item())
    return np.array(scores, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--probe", required=True, help="Concept vector .npy")
    ap.add_argument("--out_dir", default="runs/latest")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    tok_name = cfg.get("tokenizer_name", cfg["model_name"])
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    model = AutoModel.from_pretrained(
        tok_name, torch_dtype=torch.bfloat16).to("cuda").eval()

    v = np.load(args.probe, allow_pickle=True)
    v = v.reshape(-1)
    v = v / (np.linalg.norm(v) + 1e-9)

    N = _read_texts(cfg["data"]["neutrals"])
    # using anchors as hazard texts (behavior descriptions)
    H = _read_texts(cfg["data"]["anchors"])

    layer = cfg["drift"]["layer"]
    sN = drift_score(model, tok, N, v, layer)
    sH = drift_score(model, tok, H, v, layer)

    np.save(os.path.join(args.out_dir, "drift_neutral.npy"), sN)
    np.save(os.path.join(args.out_dir, "drift_hazard.npy"), sH)

    muN, varN = float(np.mean(sN)) if len(sN) else 0.0, float(np.var(sN)) if len(sN) else 0.0
    muH, varH = float(np.mean(sH)) if len(sH) else 0.0, float(np.var(sH)) if len(sH) else 0.0
    gap = muH - muN
    print(f"Neutral drift mean: {muN:.6f}")
    print(f"Neutral drift var:  {varN:.6f}")
    print(f"Hazard  drift mean: {muH:.6f}")
    print(f"Hazard  drift var:  {varH:.6f}")
    print(f"Drift gap (H-N):    {gap:.6f}")


if __name__ == "__main__":
    main()
