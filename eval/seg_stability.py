# eval/seg_stability.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Segmentation Stability Evaluation (1-edit protocol)

What this measures
------------------
Given a tokenizer and a set of texts, we:
1) Compute token-boundary sets for each original text.
2) Generate 1-edit perturbations (insert / delete / swap) with fixed rules.
3) Re-tokenize perturbed texts and compare boundary sets to the original:
   - Jaccard(B0, Bi) = |B0 ∩ Bi| / |B0 ∪ Bi|
   - Boundary flip rate = 1 - Jaccard(B0, Bi)
   - Seg-change indicator = 1{B0 != Bi}

We aggregate per-text across all perturbations and then across the corpus.
Optionally, we return 95% bootstrap CIs for the means.

Design (OOP + small patterns)
-----------------------------
- TokenBoundaryExtractor: extracts character-level boundary sets with offsets.
- OneEditPerturber: generates 1-edit variants deterministically (seeded).
- StabilityEvaluator: orchestrates evaluation & statistics.
- Bootstrapper: reusable CI computation for means and rates.

Definitions
----------
- "Boundary": character index where a token begins (0-indexed).
  We ignore special tokens; offsets come from fast tokenizers' offset mappings.

CLI
---
python -m eval.seg_stability \
  --tokenizer EleutherAI/pythia-410m \
  --texts data/eval/benign_1500.jsonl \
  --key text \
  --max_texts 400 \
  --ops insert delete swap \
  --samples_per_op 3 \
  --bootstrap 800 \
  --seed 

Outputs
-------
- Prints mean Jaccard, boundary-flip rate, and seg-change rate with (optional) CIs.
- Prints tokens-per-char (tpc) for originals (handy for your hero table).

Note
----
If offset mappings are unavailable (rare), we fall back to a whitespace+punct
heuristic to approximate boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from utils.seeding import set_global_seed


# ----------------------------- Utilities ------------------------------------


def read_jsonl_texts(path: str, key: str = "text", limit: Optional[int] = None) -> List[str]:
    """Read JSONL and return list of field `key` (non-empty)."""
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            try:
                j = json.loads(line)
            except Exception:
                continue
            t = (j.get(key) or "").strip()
            if t:
                out.append(t)
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 800, alpha: float = 0.05, seed: int = 9172) -> Tuple[float, float]:
    """Non-parametric bootstrap CI for the mean of `values`."""
    if len(values) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(values[idx].mean())
    lo, hi = np.percentile(means, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ---------------------- Token boundary extraction ----------------------------


@dataclass
class TokenBoundaryExtractor:
    """
    Extract token-boundary sets using fast tokenizer offset mappings.

    Boundary definition: character indices where a token begins (0-indexed).
    Special tokens are ignored. If offsets aren't available, we fallback to
    a whitespace-and-punctuation heuristic.

    Methods
    -------
    boundaries(text): returns a sorted tuple of boundary indices
    tokens_per_char(text): returns len(tokens)/len(text) ignoring specials
    """

    tokenizer: PreTrainedTokenizerBase

    def _fast_offsets(self, text: str) -> Optional[List[Tuple[int, int]]]:
        try:
            enc = self.tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=False,
            )
            # HF can return list-of-lists; we handle the first (single input)
            offsets = enc.get("offset_mapping")
            if isinstance(offsets, list) and len(offsets) > 0 and isinstance(offsets[0], tuple):
                return offsets  # already a flat list of tuples
            if isinstance(offsets, list) and len(offsets) > 0 and isinstance(offsets[0], list):
                return [tuple(x) for x in offsets[0]]
        except Exception:
            return None
        return None

    def _heuristic_boundaries(self, text: str) -> Tuple[int, ...]:
        # Fallback: split on whitespace/punct and reconstruct starts
        bounds: List[int] = []
        i = 0
        prev_split = True
        for idx, ch in enumerate(text):
            split_here = bool(re.match(r"\s|[^\w]", ch))
            if prev_split and not split_here:
                bounds.append(idx)
            prev_split = split_here
        return tuple(sorted(set(bounds)))

    def boundaries(self, text: str) -> Tuple[int, ...]:
        offsets = self._fast_offsets(text)
        if offsets:
            # Token starts are offsets where (start != end) (skip empty pieces)
            starts = [s for (s, e) in offsets if (e - s) > 0]
            return tuple(sorted(set(starts)))
        return self._heuristic_boundaries(text)

    def tokens_per_char(self, text: str) -> float:
        if not text:
            return 0.0
        offsets = self._fast_offsets(text)
        if offsets:
            n_tokens = sum(1 for (s, e) in offsets if (e - s) > 0)
        else:
            # heuristic token count
            n_tokens = len([w for w in re.split(r"\s+|[^\w]", text) if w])
        return float(n_tokens) / max(len(text), 1)


# ------------------------ 1-edit perturbation rules --------------------------


@dataclass
class OneEditPerturber:
    """
    Generate 1-edit variants of a string with deterministic randomness.

    Operations supported:
      - insert: insert a benign character at a random position
      - delete: delete one character (skip if too short)
      - swap:   swap two adjacent non-space characters

    Use `samples_per_op` to generate multiple variants per op per string.

    Methods
    -------
    variants(text): yields (op_name, perturbed_text)
    """

    ops: Sequence[str] = ("insert", "delete", "swap")
    samples_per_op: int = 3
    seed: int = 9172

    _INSERT_CHARS = list("!?.,:;–—-’'\"()[]{}")

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def _insert(self, s: str) -> Optional[str]:
        if not s:
            return None
        pos = self._rng.randrange(0, len(s) + 1)
        ch = self._rng.choice(self._INSERT_CHARS)
        return s[:pos] + ch + s[pos:]

    def _delete(self, s: str) -> Optional[str]:
        if len(s) < 2:
            return None
        pos = self._rng.randrange(0, len(s))
        return s[:pos] + s[pos + 1:]

    def _swap(self, s: str) -> Optional[str]:
        if len(s) < 2:
            return None
        # choose pos to swap pos and pos+1; avoid spaces for cleaner edits
        trials = 0
        while trials < 10:
            pos = self._rng.randrange(0, len(s) - 1)
            if not (s[pos].isspace() or s[pos + 1].isspace()):
                return s[:pos] + s[pos + 1] + s[pos] + s[pos + 2:]
            trials += 1
        # fallback: swap anyway
        pos = self._rng.randrange(0, len(s) - 1)
        return s[:pos] + s[pos + 1] + s[pos] + s[pos + 2:]

    def variants(self, s: str) -> Iterable[Tuple[str, str]]:
        for op in self.ops:
            for _ in range(self.samples_per_op):
                if op == "insert":
                    t = self._insert(s)
                elif op == "delete":
                    t = self._delete(s)
                elif op == "swap":
                    t = self._swap(s)
                else:
                    t = None
                if t is not None and t != s:
                    yield (op, t)


# --------------------------- Evaluator ---------------------------------------


@dataclass
class StabilityStats:
    """Container for per-text and aggregate statistics."""
    jaccards: List[float]
    flips: List[float]
    seg_changed: List[int]
    tpc_original: List[float]


class StabilityEvaluator:
    """
    Orchestrate stability evaluation given a tokenizer, texts, and a perturber.

    Metrics (per-perturbation):
      Jaccard = |B0 ∩ Bi| / |B0 ∪ Bi|
      Boundary flip rate = 1 - Jaccard
      Seg-change = 1{B0 != Bi}

    Aggregation:
      - report means across all perturbations (micro-average)
      - optional bootstrap CIs for each mean
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        perturber: OneEditPerturber,
    ):
        self.tok = tokenizer
        self.boundary_extractor = TokenBoundaryExtractor(tokenizer)
        self.perturber = perturber

    @staticmethod
    def _jaccard(a: Set[int], b: Set[int]) -> float:
        if not a and not b:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        return float(inter) / max(union, 1)

    def evaluate(
        self,
        texts: Sequence[str],
        bootstrap: int = 0,
        alpha: float = 0.05,
        seed: int = 1234,
    ) -> Tuple[StabilityStats, Optional[dict]]:
        j_list: List[float] = []
        f_list: List[float] = []
        c_list: List[int] = []
        tpc_list: List[float] = []

        for s in texts:
            b0 = set(self.boundary_extractor.boundaries(s))
            tpc0 = self.boundary_extractor.tokens_per_char(s)
            tpc_list.append(tpc0)

            for _, s2 in self.perturber.variants(s):
                b1 = set(self.boundary_extractor.boundaries(s2))
                j = self._jaccard(b0, b1)
                j_list.append(j)
                f_list.append(1.0 - j)
                c_list.append(int(b0 != b1))

        stats = StabilityStats(jaccards=j_list, flips=f_list,
                               seg_changed=c_list, tpc_original=tpc_list)

        cis = None
        if bootstrap > 0:
            j_lo, j_hi = bootstrap_ci(np.array(j_list, dtype=np.float32),
                                      n_boot=bootstrap, alpha=alpha, seed=seed)
            f_lo, f_hi = bootstrap_ci(np.array(f_list, dtype=np.float32),
                                      n_boot=bootstrap, alpha=alpha, seed=seed)
            c_lo, c_hi = bootstrap_ci(np.array(c_list, dtype=np.float32),
                                      n_boot=bootstrap, alpha=alpha, seed=seed)
            tpc_arr = np.array(tpc_list, dtype=np.float32)
            t_lo, t_hi = bootstrap_ci(tpc_arr, n_boot=bootstrap, alpha=alpha, seed=seed)
            cis = {
                "jaccard": (j_lo, j_hi),
                "flip_rate": (f_lo, f_hi),
                "seg_change_rate": (c_lo, c_hi),
                "tpc_orig": (t_lo, t_hi),
            }
        return stats, cis


# ------------------------------ CLI -----------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Segmentation stability (1-edit protocol).")
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer path or model id")
    ap.add_argument("--texts", type=str, default="", help="Optional JSONL file with texts")
    ap.add_argument("--key", type=str, default="text", help="JSONL key to read")
    ap.add_argument("--max_texts", type=int, default=400,
                    help="Number of base texts to sample/evaluate")
    ap.add_argument("--ops", nargs="+", default=["insert",
                    "delete", "swap"], help="Subset of ops to use")
    ap.add_argument("--samples_per_op", type=int, default=3, help="Perturbations per op per text")
    ap.add_argument("--bootstrap", type=int, default=800,
                    help="Bootstrap samples for 95% CI (0 to disable)")
    ap.add_argument("--alpha", type=float, default=0.05, help="1 - confidence level")
    ap.add_argument("--seed", type=int, default=9172)
    args = ap.parse_args()

    set_global_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    # default small sample set if no JSONL provided
    if args.texts and os.path.exists(args.texts):
        base_texts = read_jsonl_texts(args.texts, key=args.key, limit=args.max_texts)
    else:
        base_texts = [
            "We booked a photo shoot next weekend.",
            "The keynote had a bombastic tone throughout.",
            "He made bullet points for the meeting.",
            "They celebrated the team's explosive growth.",
            "She posted a photobomb from the party.",
        ] * max(1, args.max_texts // 5)
        base_texts = base_texts[: args.max_texts]

    perturber = OneEditPerturber(
        ops=tuple(args.ops), samples_per_op=args.samples_per_op, seed=args.seed)
    evaluator = StabilityEvaluator(tokenizer=tok, perturber=perturber)

    stats, cis = evaluator.evaluate(
        base_texts, bootstrap=args.bootstrap, alpha=args.alpha, seed=args.seed)

    j_mean = float(np.mean(stats.jaccards)) if stats.jaccards else float("nan")
    f_mean = float(np.mean(stats.flips)) if stats.flips else float("nan")
    c_mean = float(np.mean(stats.seg_changed)) if stats.seg_changed else float("nan")
    tpc_mean = float(np.mean(stats.tpc_original)) if stats.tpc_original else float("nan")

    print("=== Segmentation Stability (1-edit) ===")
    print(f"Texts: {len(base_texts)} | Ops: {','.join(args.ops)} | Samples/op: {args.samples_per_op}")
    print(f"Original TPC (mean):          {tpc_mean:6.4f}" + (
        f"   (95% CI {cis['tpc_orig'][0]:.4f}–{cis['tpc_orig'][1]:.4f})" if cis else ""))
    print(f"Jaccard (mean):               {j_mean:6.4f}" +
          (f"   (95% CI {cis['jaccard'][0]:.4f}–{cis['jaccard'][1]:.4f})" if cis else ""))
    print(f"Boundary flip rate (mean):    {f_mean:6.4f}" + (
        f"   (95% CI {cis['flip_rate'][0]:.4f}–{cis['flip_rate'][1]:.4f})" if cis else ""))
    print(f"Segmentation changed (rate):  {c_mean:6.4f}" + (
        f"   (95% CI {cis['seg_change_rate'][0]:.4f}–{cis['seg_change_rate'][1]:.4f})" if cis else ""))
    print("Notes: Boundary = character index where a token starts; specials ignored.")
    print("=======================================")


if __name__ == "__main__":
    main()
