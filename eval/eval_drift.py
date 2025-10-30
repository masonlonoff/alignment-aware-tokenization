#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
def drift_score(model: AutoModel, tok, texts: List[str], v: np.ndarray, layer: int) -> np.ndarray:
    v = torch.tensor(v, dtype=torch.float32, device=model.device).reshape(-1)
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        h = model(**ids, output_hidden_states=True).hidden_states[layer][0].mean(0)
        out.append(torch.dot(h, v).item())
    return np.array(out, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--probe", required=True, help="Concept vector .npy")
    ap.add_argument("--out_dir", default="runs/latest")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModel.from_pretrained(
        cfg["model_name"], dtype=torch.bfloat16).to("cuda").eval()

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
