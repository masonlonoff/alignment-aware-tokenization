#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Post-training sanity checks for hazard-aware SentencePiece models.

Metrics:
  1) Hazard Whole-Word Coverage:
       - % of hazard words that appear as a single token
         (for both 'word' and '▁word' forms).
  2) Benign Spillover Rate:
       - Across neutral (benign) texts, fraction of emitted pieces
         that contain any hazard stem (substring match).
  3) Piece Overlap Count:
       - Count of vocabulary entries that contain hazard stems.

Usage:
  python -m tokenizers.spm_sanity \
    --model tokenizers/spm_hazard.model \
    --vocab tokenizers/spm_hazard.vocab \
    --anchors data/anchors.jsonl \
    --neutrals data/neutral_lookalikes.jsonl \
    --min_stem_len 3 \
    --neutrals_sample 2000
"""

from __future__ import annotations
import argparse
import json
import re
from typing import List, Tuple, Set

import sentencepiece as spm


def read_jsonl_texts(path: str, key: str = "text", limit: int | None = None) -> List[str]:
    """Read a JSONL file and return a list of strings from field `key`."""
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            try:
                j = json.loads(line)
                t = j.get(key, "")
                if isinstance(t, str) and t.strip():
                    out.append(t.strip())
            except Exception:
                continue
    return out


def extract_hazard_words_and_stems(anchors: List[str], min_stem_len: int = 3) -> Tuple[Set[str], Set[str]]:
    """From hazard anchors, get unique whole words and stems (lower, alnum)."""
    words, stems = set(), set()
    for t in anchors:
        toks = re.split(r"\s+|[^\w]", t.lower())
        for w in toks:
            w = "".join(ch for ch in w if ch.isalnum())
            if not w:
                continue
            words.add(w)
            if len(w) >= min_stem_len:
                stems.add(w)
    return words, stems


def hazard_coverage(sp: spm.SentencePieceProcessor, hazard_words: Set[str]) -> Tuple[float, float]:
    """
    % of hazard words segmented as a single piece.

    Returns:
        (plain_coverage, word_begin_coverage)
        - plain_coverage: word as single piece (no leading '▁')
        - word_begin_coverage: '▁word' as single piece
    """
    plain_hits = wb_hits = 0
    for w in hazard_words:
        # plain
        plain = sp.EncodeAsPieces(w)
        if len(plain) == 1:
            plain_hits += 1
        # word-begin ('▁' is space marker)
        wb = sp.EncodeAsPieces(" " + w)  # ensures a word-begin boundary
        if len(wb) == 2 and wb[0] == "▁":
            # segmentation like: ['▁', 'word'] (unlikely); prefer single piece
            pass
        if len(wb) == 1 and wb[0].startswith("▁"):
            wb_hits += 1
    n = max(len(hazard_words), 1)
    return 100.0 * plain_hits / n, 100.0 * wb_hits / n


def benign_spillover_rate(sp: spm.SentencePieceProcessor, neutrals: List[str], hazard_stems: Set[str], sample: int) -> float:
    """
    Fraction of emitted pieces in neutral texts that contain any hazard stem.
    """
    from itertools import islice
    texts = list(islice(neutrals, 0, min(sample, len(neutrals))))
    total, risky = 0, 0
    for t in texts:
        pieces = sp.EncodeAsPieces(t)
        for p in pieces:
            total += 1
            p0 = p.replace("▁", "")
            if any(s in p0 for s in hazard_stems):
                risky += 1
    return 100.0 * (risky / max(total, 1))


def piece_overlap_count(vocab_path: str, hazard_stems: Set[str]) -> int:
    """
    Count how many vocab entries contain any hazard stem (substring).
    """
    count = 0
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            piece = line.split("\t", 1)[0].strip()
            p0 = piece.replace("▁", "")
            if any(s in p0 for s in hazard_stems):
                count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="SentencePiece .model file")
    ap.add_argument("--vocab", required=True, help="SentencePiece .vocab file")
    ap.add_argument("--anchors", required=True, help="JSONL with {'text': ...} hazard anchors")
    ap.add_argument("--neutrals", required=True,
                    help="JSONL with {'text': ...} neutral look-alikes")
    ap.add_argument("--min_stem_len", type=int, default=3)
    ap.add_argument("--neutrals_sample", type=int, default=2000)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.model)
    anchors = read_jsonl_texts(args.anchors)
    neutrals = read_jsonl_texts(args.neutrals)
    hazard_words, hazard_stems = extract_hazard_words_and_stems(anchors, args.min_stem_len)

    plain_cov, wb_cov = hazard_coverage(sp, hazard_words)
    spill = benign_spillover_rate(sp, neutrals, hazard_stems, args.neutrals_sample)
    overlap_ct = piece_overlap_count(args.vocab, hazard_stems)

    print("=== SPM Hazard Sanity Check ===")
    print(f"Hazard single-piece coverage (plain):     {plain_cov:6.2f}%")
    print(f"Hazard single-piece coverage (▁word):     {wb_cov:6.2f}%")
    print(f"Benign spillover rate (piece-level):      {spill:6.2f}%")
    print(f"Vocab entries overlapping hazard stems:   {overlap_ct:d}")
    print("================================")
    # You can add thresholds/returns here to fail CI if needed.


if __name__ == "__main__":
    main()
