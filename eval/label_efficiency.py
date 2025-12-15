#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Label-efficiency: train a tiny classifier on frozen mid-layer features for H vs N
with budgets {50,100,300}; report F1 and AUPRC as mean±std across repeated trials.

Fixes vs previous version:
- Stratified train/test split
- Multiple subsampling trials per budget (variance-aware)
- No re-seeding inside the loop (stable but not degenerate)
- Optional duplicate/leakage check by exact text overlap
"""
from __future__ import annotations

import argparse
import json
from typing import List, Tuple

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel


def _read_texts(path: str) -> List[str]:
    xs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            t = (j.get("text") or "").strip()
            if t:
                xs.append(t)
    return xs


def _dedup_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _overlap_count(a: List[str], b: List[str]) -> int:
    sa = set(a)
    sb = set(b)
    return len(sa.intersection(sb))


@torch.no_grad()
def pooled_hidden(model, tok, texts: List[str], layer: int, max_len: int = 256) -> np.ndarray:
    """
    Mean-pool hidden states at a given layer across sequence positions.
    Returns float32 numpy array: [n_texts, hidden_dim].
    """
    device = model.device

    # ensure tokenizer has a pad token
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "right"

    embs: List[np.ndarray] = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer][0].mean(0).to(torch.float32)  # [hid]
        embs.append(h.cpu().numpy())

    hidden_size = getattr(model.config, "hidden_size", embs[0].shape[0] if embs else 0)
    return np.stack(embs) if embs else np.zeros((0, hidden_size), dtype=np.float32)


def _fit_eval_once(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Train on a random subset of size k from training set, evaluate on test.
    Returns (F1, AUPRC).
    """
    k_eff = min(k, len(Xtr))
    sub_idx = rng.choice(len(Xtr), size=k_eff, replace=False)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    ).fit(Xtr[sub_idx], ytr[sub_idx])

    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    f1 = f1_score(yte, pred)
    auprc = average_precision_score(yte, prob)
    return float(f1), float(auprc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--budgets", nargs="+", type=int, default=[50, 100, 300])
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--trials", type=int, default=10, help="Subsampling trials per budget")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument(
        "--dedup",
        action="store_true",
        help="Deduplicate exact text strings within H and N before embedding",
    )
    ap.add_argument(
        "--check_overlap",
        action="store_true",
        help="Check exact train/test text overlap (leakage diagnostic)",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_name = cfg["model_name"]
    layer = int(cfg["drift"]["layer"])

    # Load data
    H = _read_texts(cfg["data"]["anchors"])
    N = _read_texts(cfg["data"]["neutrals"])

    if args.dedup:
        H = _dedup_keep_order(H)
        N = _dedup_keep_order(N)

    print(f"[data] |H|={len(H)}  |N|={len(N)}  layer={layer}")

    # Load model
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()

    # Embed
    XH = pooled_hidden(model, tok, H, layer=layer, max_len=args.max_len)
    XN = pooled_hidden(model, tok, N, layer=layer, max_len=args.max_len)

    yH = np.ones(len(XH), dtype=np.int64)
    yN = np.zeros(len(XN), dtype=np.int64)
    X = np.vstack([XH, XN])
    y = np.concatenate([yH, yN])

    # Keep text list aligned to X for overlap diagnostics
    texts = H + N

    # Stratified split
    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
        shuffle=True,
    )

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    if args.check_overlap:
        tr_texts = [texts[i] for i in tr_idx]
        te_texts = [texts[i] for i in te_idx]
        ov = _overlap_count(tr_texts, te_texts)
        print(f"[leakage] exact train/test text overlap: {ov} items")

    print(f"[split] train={len(Xtr)} test={len(Xte)}  pos_train={ytr.mean():.3f} pos_test={yte.mean():.3f}")

    rng = np.random.default_rng(args.seed)  # ONE rng for everything

    # Evaluate label-efficiency with variance
    for k in args.budgets:
        f1s, auprcs = [], []
        for _ in range(args.trials):
            f1, auprc = _fit_eval_once(Xtr, ytr, Xte, yte, k=k, rng=rng)
            f1s.append(f1)
            auprcs.append(auprc)

        f1_mean, f1_std = float(np.mean(f1s)), float(np.std(f1s))
        ap_mean, ap_std = float(np.mean(auprcs)), float(np.std(auprcs))

        print(
            f"labels={min(k, len(Xtr)):<4d} "
            f"F1={f1_mean:.4f}±{f1_std:.4f}  "
            f"AUPRC={ap_mean:.4f}±{ap_std:.4f}  "
            f"(trials={args.trials})"
        )


if __name__ == "__main__":
    main()
