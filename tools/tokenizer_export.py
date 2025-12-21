#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert a SentencePiece .model (Unigram) to a Hugging Face tokenizer.json
with correct special tokens. Optionally inherit special tokens from a base HF
model (recommended: use the LM you plan to fine-tune, e.g., LLaMA/Mistral/Qwen).

Typical usage:
  python -m tools.tokenizer_export \
    --spm_model tokenizers/spm_hazard/spm_mistral7b_hazard.model \
    --out_dir tokenizers/spm_hazard/hf_mistral7b_hazard \
    --base_model mistralai/Mistral-7B-v0.1

Outputs:
  - <out_dir>/tokenizer.json
  - <out_dir>/tokenizer_config.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from tokenizers import normalizers, pre_tokenizers, decoders, Tokenizer
from tokenizers.models import Unigram
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from sentencepiece import sentencepiece_model_pb2 as spmp


def _load_specials_from_base(base_model: str) -> dict:
    """Pull special token strings from a base HF tokenizer if available."""
    base = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    specials = {
        "bos_token": getattr(base, "bos_token", None),
        "eos_token": getattr(base, "eos_token", None),
        "unk_token": getattr(base, "unk_token", "<unk>"),
        "pad_token": getattr(base, "pad_token", None),
    }
    # Remove Nones (PreTrainedTokenizerFast handles missing values)
    return {k: v for k, v in specials.items() if v is not None}


def _default_specials() -> dict:
    """Sensible defaults for Unigram/SentencePiece-style tokenizers."""
    return {
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
    }


def spm_to_hf(spm_model_path: str, out_dir: str, base_model: Optional[str] = None) -> str:
    """
    Load a SentencePiece Unigram model and export as a HF PreTrainedTokenizerFast.

    Args:
        spm_model_path: path to .model (trained with spm_priors or vanilla SPM)
        out_dir: directory to write tokenizer.json + tokenizer_config.json
        base_model: optional HF model to copy special tokens from

    Returns:
        out_dir
    """
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1) Read the SentencePiece .model via the official proto
    # -------------------------------------------------------------
    proto = spmp.ModelProto()
    with open(spm_model_path, "rb") as f:
        proto.ParseFromString(f.read())

    # Build vocab list: [(piece, score), ...]
    vocab = [(p.piece, p.score) for p in proto.pieces]

    # Find unk_id from the SentencePiece proto
    unk_id = None
    for i, p in enumerate(proto.pieces):
        if (
            p.type == spmp.ModelProto.SentencePiece.Type.UNKNOWN
            or p.piece == "<unk>"
        ):
            unk_id = i
            break

    # Fallback: assume first piece is unk if nothing was explicitly marked
    if unk_id is None:
        unk_id = 0

    # Create Unigram model with a valid unk_id
    model = Unigram(vocab, unk_id=unk_id)
    tok: Tokenizer = Tokenizer(model)

    # -------------------------------------------------------------
    # 2) Normalizer / pre-tokenizer / decoder (SPM-like)
    # -------------------------------------------------------------
    tok.normalizer = normalizers.Sequence([normalizers.NFKC()])

    # Metaspace API changed across tokenizers versions; be defensive.
    try:
        metaspace_pretok = pre_tokenizers.Metaspace(
            replacement="▁", add_prefix_space=True
        )
    except TypeError:
        metaspace_pretok = pre_tokenizers.Metaspace(replacement="▁")

    try:
        metaspace_dec = decoders.Metaspace(
            replacement="▁", add_prefix_space=True
        )
    except TypeError:
        metaspace_dec = decoders.Metaspace(replacement="▁")

    tok.pre_tokenizer = pre_tokenizers.Sequence([metaspace_pretok])
    tok.decoder = metaspace_dec

    # -------------------------------------------------------------
    # 3) Wrap with HF fast tokenizer and save
    # -------------------------------------------------------------
    specials = _load_specials_from_base(base_model) if base_model else _default_specials()

    hf_fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        **specials,
    )

    hf_fast.save_pretrained(out_dir)

    # Patch tokenizer_config.json with a couple of handy fields
    cfg_path = os.path.join(out_dir, "tokenizer_config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
    else:
        cfg = {}
    cfg.update(
        {
            "model_max_length": 4096,
            "tokenizer_class": "PreTrainedTokenizerFast",
        }
    )
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"[export] Wrote HF tokenizer to: {out_dir}")
    print(f"         Specials: {specials}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spm_model", required=True, help="Path to SentencePiece .model")
    ap.add_argument("--out_dir", required=True, help="Directory to write tokenizer.json")
    ap.add_argument("--base_model", default=None, help="HF model to copy special tokens from")
    args = ap.parse_args()

    spm_to_hf(args.spm_model, args.out_dir, base_model=args.base_model)


if __name__ == "__main__":
    main()
