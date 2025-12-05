#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hazard-aware SentencePiece (Unigram) training with token-level priors.

This module implements method (C) from the proposal:
  - Train a SentencePiece Unigram tokenizer with *hazard-aware priors* that
    (i) penalize pieces likely to cause subword spillover/overlap with hazard stems,
    (ii) boost whole-word hazard tokens so they are segmented as single pieces.
  - Works by supplying a *seed sentencepieces* file with scores, which SentencePiece
    uses as initialization for its EM objective (a practical way to inject priors).

Design:
  - Strategy:       PriorStrategy (computes token penalties/boosts)
                    CandidateFilter (optional pre/post filtering of risky pieces)
  - Template/Facade: SPMHazardTrainer orchestrates data prep → seed build → Train

Key ideas:
  - overlap(token): does token contain (substring-level) any hazard stem?
  - spillover(token): how often token substrings appear across benign neighbors?
  - penalty(token) = λ_conf * spillover + λ_overlap * overlap
  - For hazard whole words: apply positive *boost* so they become stable whole tokens.
  - We encode "word-begin" variants via '▁' (SentencePiece whitespace marker).

Usage:
  python -m tokenizers.spm_priors \
    --corpus data/unlabeled.txt \
    --anchors data/anchors.jsonl \
    --neutrals data/neutral_lookalikes.jsonl \
    --model_prefix tokenizers/spm_hazard \
    --vocab_size 50000 \
    --lambda_conf 0.3 \
    --lambda_overlap 0.5 \
    --hazard_boost 5.0 \
    --min_stem_len 3 \
    --sample_size 200000 \
    --byte_fallback true

Outputs:
  - <model_prefix>.model / <model_prefix>.vocab (standard SentencePiece artifacts)
  - A companion seed file (<model_prefix>_seed.txt) for reproducibility.
  - A tiny JSON log (<model_prefix>_hazard_log.json) with lexicon/prior/training summary.
"""

from __future__ import annotations
import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import sentencepiece as spm


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def read_jsonl_texts(path: str, key: str = "text", limit: int | None = None) -> List[str]:
    """
    Read a JSONL file and return a list of strings from field `key`.

    Args:
        path:  Path to .jsonl file (each line is a JSON object).
        key:   Field holding the string to extract.
        limit: If set, stop after this many lines.

    Returns:
        List of non-empty strings.
    """
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


def tokenize_words(texts: Sequence[str]) -> List[str]:
    """
    Very lightweight whitespace/ punct splitting to obtain words.

    Args:
        texts: Documents/sentences.

    Returns:
        Flat list of lowercase "words" (alnum-only).
    """
    words: List[str] = []
    for t in texts:
        for w in re.split(r"\s+|[^\w]", t.lower()):
            w = "".join(ch for ch in w if ch.isalnum())
            if w:
                words.append(w)
    return words


def unique(seq: Iterable[str]) -> List[str]:
    """Stable unique preserving order."""
    seen = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def apply_hf_preset(args) -> None:
    """
    Mutates args with sensible defaults per family if not explicitly set.
    Keeps user overrides if provided.
    """
    if args.hf_target == "none":
        return

    # Common SPM defaults for these families
    if not args.normalization_rule_name or args.normalization_rule_name == "nmt":
        args.normalization_rule_name = "nmt"
    # Strong byte fallback is pragmatic for robustness
    if not args.byte_fallback:
        args.byte_fallback = "true"
    # High coverage to keep rare chars
    if not args.character_coverage:
        args.character_coverage = 0.9995

    # Special tokens presets are conservative—override via explicit flags if needed.
    presets = {
        "llama3":    {"bos": "<s>", "eos": "</s>", "pad": "", "extra": []},
        "mistral7b": {"bos": "<s>", "eos": "</s>", "pad": "", "extra": []},
        "qwen2_7b":  {"bos": "<s>", "eos": "</s>", "pad": "", "extra": []},
    }
    sp = presets.get(args.hf_target, {"bos": "", "eos": "", "pad": "", "extra": []})

    # Only set if user didn’t supply explicit tokens
    args.bos_token = args.bos_token or sp["bos"]
    args.eos_token = args.eos_token or sp["eos"]
    args.pad_token = args.pad_token or sp["pad"]

    if args.user_defined_symbols == "" and args.addl_special == "":
        # leave empty unless user passes addl_special
        pass


# ---------------------------------------------------------------------
# Data model & strategies
# ---------------------------------------------------------------------

@dataclass
class HazardLexicon:
    """
    Holds hazard stems and benign neighbor word counts for spillover heuristics.

    Attributes:
        stems:           Sorted list of hazard stems (alnum, lowercased).
        benign_counts:   Map stem -> number of neutral examples containing it.
        hazard_words:    Whole words (from anchors) to optionally boost/pin.
    """
    stems: List[str]
    benign_counts: Dict[str, int]
    hazard_words: List[str]


def extract_hazard_lexicon(anchors: Sequence[str],
                           neutrals: Sequence[str],
                           min_stem_len: int = 3,
                           max_hits_per_stem: int = 1000) -> HazardLexicon:
    """
    Build a minimal hazard lexicon from few-shot anchors and neutral look-alikes.

    - Stems are constructed by alnum-lowering each token and filtering short units.
    - benign_counts counts how often each stem appears inside neutral texts.

    Args:
        anchors: Hazard anchor texts.
        neutrals: Neutral look-alikes (benign neighbors).
        min_stem_len: Minimum length for stems.
        max_hits_per_stem: Early stop per stem when scanning neutrals.

    Returns:
        HazardLexicon dataclass.
    """
    # Hazard stems
    stems_set = set()
    hazard_words_set = set()
    for t in anchors:
        toks = ["".join(ch for ch in w.lower() if ch.isalnum()) for w in t.split()]
        toks = [w for w in toks if w]
        hazard_words_set.update(toks)
        for w in toks:
            if len(w) >= min_stem_len:
                stems_set.add(w)
    stems = sorted(stems_set)
    hazard_words = sorted(hazard_words_set)

    # Count neutral co-occurrences per stem
    counts: Dict[str, int] = {s: 0 for s in stems}
    for s in stems:
        s_low = s
        hits = 0
        for text in neutrals:
            if s_low in text.lower():
                hits += 1
                if hits >= max_hits_per_stem:
                    break
        counts[s] = hits

    return HazardLexicon(stems=stems, benign_counts=counts, hazard_words=hazard_words)


class PriorStrategy:
    """
    Strategy base for computing prior-adjusted scores for seed sentencepieces.

    SentencePiece accepts a seed file with lines: "<piece>\t<score>".
    Higher scores mean higher prior likelihood of inclusion.
    """

    def score(self, piece: str) -> float:
        raise NotImplementedError


@dataclass
class HazardAwarePrior(PriorStrategy):
    """
    Hazard-aware prior using overlap & spillover proxies.

    Scoring:
        base_score(piece) - λ_conf * spillover(piece) - λ_overlap * overlap(piece)
      + optional boost for hazard *whole words* (with and without '▁').

    Heuristics:
      - overlap(piece) = 1 if piece contains any hazard stem (substring), else 0
      - spillover(piece) ~ normalized count of how many neutral examples contain
        that stem (max over stems present in piece). This approximates the risk
        of a piece appearing in benign look-alikes.

    Args:
        lex:                HazardLexicon (stems, counts, hazard words).
        lambda_conf:        Coefficient for spillover penalty.
        lambda_overlap:     Coefficient for overlap penalty.
        hazard_boost:       Positive boost for whole hazard words.
        base_score:         Baseline log-score assigned to most pieces.
        normalize_counts:   Divide stem counts by `count_norm` before scoring.
        count_norm:         Normalization constant for benign_counts.
    """
    lex: HazardLexicon
    lambda_conf: float = 0.3
    lambda_overlap: float = 0.5
    hazard_boost: float = 5.0
    base_score: float = -1.0
    normalize_counts: bool = True
    count_norm: float = 100.0

    def _overlap(self, piece: str) -> int:
        p = piece.replace("▁", "")  # ignore leading-space marker for overlap calc
        for s in self.lex.stems:
            if s in p:
                return 1
        return 0

    def _spillover(self, piece: str) -> float:
        p = piece.replace("▁", "")
        val = 0.0
        for s in self.lex.stems:
            if s in p:
                c = float(self.lex.benign_counts.get(s, 0))
                if self.normalize_counts and self.count_norm > 0:
                    c = min(c / self.count_norm, 1.0)
                val = max(val, c)  # conservative: take worst offending stem
        return val

    def _hazard_whole_word_boost(self, piece: str) -> float:
        p = piece.replace("▁", "")
        return self.hazard_boost if p in self.lex.hazard_words else 0.0

    def score(self, piece: str) -> float:
        """
        Return a prior score for `piece` (higher = keep as candidate).

        Special-cases:
          - Give positive boost to hazard *whole words* (word-begin variant '▁word'
            AND plain 'word'), which nudges SPM to keep them as single tokens.
          - Penalize pieces likely to cause spillover.

        Returns:
            A float log-score compatible with SentencePiece seed format.
        """
        score = self.base_score
        score -= self.lambda_conf * self._spillover(piece)
        score -= self.lambda_overlap * self._overlap(piece)
        score += self._hazard_whole_word_boost(piece)
        return float(score)


class CandidateFilter:
    """
    Optional strategy to drop or retain pieces before EM.

    Implementations can hard-remove clearly risky pieces, or keep a whitelist.
    """

    def keep(self, piece: str) -> bool:
        raise NotImplementedError


class KeepAllFilter(CandidateFilter):
    """Default filter that retains every piece."""

    def keep(self, piece: str) -> bool:
        return True


class DropRiskySubstrings(CandidateFilter):
    """
    Simple filter: drop pieces that are short substrings of hazard stems.

    Args:
        lex:         HazardLexicon
        min_len:     Ignore substrings shorter than this (often noise).
        drop_if_in:  If True, drop if piece (minus '▁') is found in any stem.
    """

    def __init__(self, lex: HazardLexicon, min_len: int = 2, drop_if_in: bool = True):
        self.lex = lex
        self.min_len = min_len
        self.drop_if_in = drop_if_in

    def keep(self, piece: str) -> bool:
        p = piece.replace("▁", "")
        if len(p) < self.min_len:
            return True
        if self.drop_if_in:
            for s in self.lex.stems:
                if p and p in s and p != s:
                    # Only drop small substrings (e.g., 'bo' in 'bomb')
                    if len(p) <= max(3, len(s) // 2):
                        return False
        return True


# ---------------------------------------------------------------------
# Seed builder and trainer orchestrator
# ---------------------------------------------------------------------

def make_seed_sentencepieces(lex: HazardLexicon,
                             prior: PriorStrategy,
                             extra_seed: Sequence[str] | None = None,
                             include_hazard_whole_words: bool = True) -> List[Tuple[str, float]]:
    """
    Construct seed sentencepieces list (piece, score).

    We include:
      - hazard whole words (with and without leading '▁') if requested;
      - optionally extra_seed items (e.g., hand-picked safe pieces);
      - We DO NOT enumerate arbitrary substrings here (SPM will do that); we
        supply priors only for pieces we care to bias.

    Args:
        lex: Hazard lexicon.
        prior: PriorStrategy to score pieces.
        extra_seed: Optional iterable of additional seed pieces to score.
        include_hazard_whole_words: If True, add hazard words and '▁'+word.

    Returns:
        List of (piece, score) sorted by score (desc).
    """
    seeds: List[str] = []
    if include_hazard_whole_words:
        for w in lex.hazard_words:
            seeds.append(w)
            seeds.append("▁" + w)

    if extra_seed:
        seeds.extend(extra_seed)

    # unique and score
    seeds = unique(seeds)
    scored = [(p, float(prior.score(p))) for p in seeds]
    # sort (SPM doesn't require, but useful for debugging)
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def write_seed_file(seed: List[Tuple[str, float]], path: str) -> None:
    """
    Write a seed sentencepieces text file with "<piece>\t<score>" per line.

    Args:
        seed: List of (piece, score).
        path: Destination path (will overwrite).
    """
    with open(path, "w", encoding="utf-8") as f:
        for p, s in seed:
            # SentencePiece allows comments with '#'; we keep plain lines.
            f.write(f"{p}\t{s:.6f}\n")


@dataclass
class SPMConfig:
    """
    SentencePiece training configuration subset for our hazard-aware Unigram.

    Attributes map to the CLI flags of SentencePieceTrainer.
    """
    model_prefix: str
    vocab_size: int
    corpus: str
    model_type: str = "unigram"
    byte_fallback: bool = True
    normalization_rule_name: str = "nmt"
    input_sentence_size: int = 200000
    shuffle_input_sentence: bool = True
    character_coverage: float = 0.9995
    hard_vocab_limit: bool = True  # if False, can exceed vocab_size slightly
    user_defined_symbols: List[str] | None = None  # e.g., control tokens


class SPMHazardTrainer:
    """
    Orchestrates *hazard-aware Unigram* training via seed priors.

    Template:
      1) Load anchors & neutrals → build HazardLexicon
      2) Build PriorStrategy & CandidateFilter
      3) Construct a seed sentencepieces file with adjusted scores
      4) Call SentencePieceTrainer.Train(...) with --seed_sentencepieces=...
      5) (Optional) Post sanity checks / logging

    The result is a standard <model_prefix>.model / .vocab pair usable by HF.
    """

    def __init__(self,
                 cfg: SPMConfig,
                 prior: PriorStrategy,
                 candidate_filter: CandidateFilter | None = None):
        self.cfg = cfg
        self.prior = prior
        self.candidate_filter = candidate_filter or KeepAllFilter()

    def _format_user_defined(self) -> str:
        """Comma-join user-defined symbols for SPM arg (empty string if none)."""
        uds = self.cfg.user_defined_symbols or []
        return ",".join(uds)

    def _jsonl_to_txt(self, in_path: str, key: str = "text") -> str:
        """If input looks like JSONL, convert to a temp TXT (one-per-line) for SPM."""
        if not in_path.lower().endswith(".jsonl"):
            return in_path
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        tmp_path = tmp.name
        tmp.close()
        xs = read_jsonl_texts(in_path, key=key)
        with open(tmp_path, "w", encoding="utf-8") as f:
            for t in xs:
                f.write(t + "\n")
        return tmp_path

    def _rescore_model(self, model_path: str, prior: PriorStrategy) -> int:
        """
        Post-EM hazard-aware rescore: open .model proto and replace each piece's
        score with the prior-adjusted score. Returns number of pieces changed.
        """
        from sentencepiece import sentencepiece_model_pb2 as spmp
        proto = spmp.ModelProto()
        with open(model_path, "rb") as f:
            proto.ParseFromString(f.read())

        changed = 0
        for p in proto.pieces:
            if p.type in (
                spmp.ModelProto.SentencePiece.Type.UNKNOWN,
                spmp.ModelProto.SentencePiece.Type.CONTROL,
                spmp.ModelProto.SentencePiece.Type.USER_DEFINED,
                spmp.ModelProto.SentencePiece.Type.BYTE,
            ):
                continue
            try:
                new_score = float(prior.score(p.piece))
            except Exception:
                continue
            if abs(new_score - p.score) > 1e-9:
                p.score = new_score
                changed += 1

        with open(model_path, "wb") as f:
            f.write(proto.SerializeToString())
        return changed

    def train(self,
              anchors_path: str,
              neutrals_path: str,
              min_stem_len: int = 3,
              extra_seed: Sequence[str] | None = None) -> Tuple[str, str, dict]:
        """
        Run hazard-aware Unigram training with:
        A) try seeded priors (if supported),
        B) else vanilla + post-EM hazard-aware rescore,
        And auto-fallback from normalization 'nmt' -> 'identity' on Windows wheels lacking 'nmt'.

        Returns:
            (model_path, vocab_path, train_log_dict)
        """
        # ---- Seed construction ----
        seed_list = make_seed_sentencepieces(
            lex=self.prior.lex, prior=self.prior,
            extra_seed=extra_seed, include_hazard_whole_words=True
        )
        filtered_seed = [
            (p, s) for (p, s) in seed_list if self.candidate_filter.keep(p)
        ]
        seed_path = f"{self.cfg.model_prefix}_seed.txt"
        write_seed_file(filtered_seed, seed_path)

        # Convert JSONL to TXT for SPM if needed
        input_path = self._jsonl_to_txt(self.cfg.corpus)

        # Base kwargs (safer than a single cmd string on Windows)
        base_kwargs = dict(
            model_prefix=self.cfg.model_prefix,
            model_type=self.cfg.model_type,
            input=input_path,
            vocab_size=int(self.cfg.vocab_size),
            normalization_rule_name=self.cfg.normalization_rule_name or "nmt",
            input_sentence_size=int(self.cfg.input_sentence_size),
            shuffle_input_sentence=bool(self.cfg.shuffle_input_sentence),
            character_coverage=float(self.cfg.character_coverage),
            hard_vocab_limit=bool(self.cfg.hard_vocab_limit),
            byte_fallback=bool(self.cfg.byte_fallback),
        )
        uds = self._format_user_defined()
        if uds:
            base_kwargs["user_defined_symbols"] = uds

        def _try_train(kwargs: dict) -> None:
            """Try SPM Train with given kwargs, else raise."""
            import copy
            spm.SentencePieceTrainer.Train(**copy.deepcopy(kwargs))

        used_seeded = False
        seeded_norm_name: str | None = None
        vanilla_norm_name: str | None = None
        rescore_changed = 0

        # ---- Attempt seeded path first ----
        try:
            seeded_kwargs = dict(base_kwargs)
            seeded_kwargs["seed_sentencepieces"] = seed_path
            seeded_norm_name = seeded_kwargs.get("normalization_rule_name", "nmt")
            _try_train(seeded_kwargs)
            used_seeded = True
            print("[spm_priors] Seeded training path used (seed_sentencepieces).")
        except OSError as e:
            msg = str(e)
            if "unknown field name" in msg and "seed_sentencepieces" in msg:
                print("[spm_priors] seed_sentencepieces unsupported; falling back to vanilla + hazard-rescore.")
            elif "No precompiled charsmap is found" in msg:
                # Fallback nmt -> identity for seeded path
                print(
                    "[spm_priors] 'nmt' charsmap missing; retrying seeded with normalization_rule_name='identity'...")
                seeded_kwargs = dict(base_kwargs)
                seeded_kwargs["seed_sentencepieces"] = seed_path
                seeded_kwargs["normalization_rule_name"] = "identity"
                seeded_norm_name = "identity"
                try:
                    _try_train(seeded_kwargs)
                    used_seeded = True
                    print("[spm_priors] Seeded training succeeded with normalization='identity'.")
                except Exception as e2:
                    print(
                        f"[spm_priors] Seeded+identity failed ({type(e2).__name__}); will try vanilla + rescore.")
            else:
                print(
                    f"[spm_priors] Seeded training failed ({type(e).__name__}: {msg[:160]}); will try vanilla + rescore."
                )
        except Exception as e:
            print(
                f"[spm_priors] Seeded training failed ({type(e).__name__}: {str(e)[:160]}); will try vanilla + rescore."
            )

        # ---- Vanilla + post-EM rescore if seeded not used ----
        if not used_seeded:
            try:
                vanilla_norm_name = base_kwargs.get("normalization_rule_name", "nmt")
                _try_train(base_kwargs)
                print("[spm_priors] Trained vanilla SPM. Applying hazard-aware post-EM rescore...")
            except OSError as e:
                msg = str(e)
                if "No precompiled charsmap is found" in msg:
                    # Fallback nmt -> identity for vanilla path
                    print(
                        "[spm_priors] 'nmt' charsmap missing; retrying vanilla with normalization_rule_name='identity'...")
                    vanilla_kwargs = dict(base_kwargs)
                    vanilla_kwargs["normalization_rule_name"] = "identity"
                    vanilla_norm_name = "identity"
                    _try_train(vanilla_kwargs)
                    print(
                        "[spm_priors] Vanilla SPM succeeded with normalization='identity'. Applying hazard-aware post-EM rescore...")
                else:
                    raise

            model_path = f"{self.cfg.model_prefix}.model"
            rescore_changed = self._rescore_model(model_path, self.prior)
            print(f"[spm_priors] Rescored {rescore_changed} pieces via hazard-aware priors.")

        model_path = f"{self.cfg.model_prefix}.model"
        vocab_path = f"{self.cfg.model_prefix}.vocab"

        # ---- Tiny JSON summary log ----
        lex = self.prior.lex
        max_benign = max(lex.benign_counts.values(), default=0)
        train_log = {
            "lexicon": {
                "num_stems": len(lex.stems),
                "num_hazard_words": len(lex.hazard_words),
                "max_benign_count": int(max_benign),
            },
            "prior": {
                "lambda_conf": float(getattr(self.prior, "lambda_conf", 0.0)),
                "lambda_overlap": float(getattr(self.prior, "lambda_overlap", 0.0)),
                "hazard_boost": float(getattr(self.prior, "hazard_boost", 0.0)),
                "base_score": float(getattr(self.prior, "base_score", 0.0)),
                "count_norm": float(getattr(self.prior, "count_norm", 1.0)),
            },
            "seed": {
                "path": seed_path,
                "num_seed_pieces": len(filtered_seed),
            },
            "training": {
                "used_seeded": bool(used_seeded),
                "seeded_normalization": seeded_norm_name,
                "vanilla_normalization": vanilla_norm_name,
                "rescore_changed_pieces": int(rescore_changed),
            },
        }

        return model_path, vocab_path, train_log


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train hazard-aware SPM (Unigram) with token-level priors.")
    p.add_argument("--corpus", required=True,
                   help="Path to training text corpus (one sentence per line, or JSONL with 'text').")
    p.add_argument("--anchors", required=True, help="JSONL with {'text': ...} hazard anchors.")
    p.add_argument("--neutrals", required=True,
                   help="JSONL with {'text': ...} neutral look-alikes.")
    p.add_argument("--model_prefix", required=True, help="Prefix for SentencePiece outputs.")
    p.add_argument("--vocab_size", type=int, default=50000)
    p.add_argument("--model_type", type=str, default="unigram")
    p.add_argument("--byte_fallback", type=str, default="true", choices=["true", "false"])
    p.add_argument("--normalization_rule_name", type=str, default="nmt")
    p.add_argument("--input_sentence_size", type=int, default=200000)
    p.add_argument("--shuffle_input_sentence", type=str, default="true", choices=["true", "false"])
    p.add_argument("--character_coverage", type=float, default=0.9995)
    p.add_argument("--hard_vocab_limit", type=str, default="true", choices=["true", "false"])
    p.add_argument("--user_defined_symbols", type=str, default="",
                   help="Comma-separated list, e.g., <s>,</s>,<pad>.")
    # Prior hyperparams
    p.add_argument("--lambda_conf", type=float, default=0.3, help="Weight for spillover penalty.")
    p.add_argument("--lambda_overlap", type=float, default=0.5, help="Weight for overlap penalty.")
    p.add_argument("--hazard_boost", type=float, default=5.0,
                   help="Positive boost for hazard whole words.")
    p.add_argument("--base_score", type=float, default=-1.0,
                   help="Default seed log-score for pieces.")
    p.add_argument("--min_stem_len", type=int, default=3)
    p.add_argument("--count_norm", type=float, default=100.0,
                   help="Normalization for benign stem counts.")
    # Filtering
    p.add_argument("--drop_risky_substrings", action="store_true",
                   help="Pre-filter short substrings of hazard stems.")
    p.add_argument("--min_substring_len", type=int, default=2)
    p.add_argument("--hf_target", type=str, default="none",
                   choices=["none", "llama3", "mistral7b", "qwen2_7b"],
                   help="Preset SPM settings + common special tokens for target family.")
    # explicit special tokens (override presets if provided)
    p.add_argument("--bos_token", type=str, default="")
    p.add_argument("--eos_token", type=str, default="")
    p.add_argument("--pad_token", type=str, default="")
    p.add_argument("--addl_special", type=str, default="",
                   help="Comma-separated extra special tokens to register.")
    # optional export to HF fast tokenizer folder
    p.add_argument("--export_hf_dir", type=str, default="",
                   help="If set, writes a HF fast tokenizer folder from the trained SPM model.")
    # Extras
    p.add_argument("--extra_seed", type=str, default="",
                   help="Path to a text file with extra seed pieces (one per line).")
    return p


def main():
    args = build_cli().parse_args()
    apply_hf_preset(args)

    # Build user_defined_symbols list from preset + explicit extras
    ud_list = []
    if args.addl_special:
        ud_list.extend([s for s in args.addl_special.split(",") if s])
    uds = [s for s in ud_list] if ud_list else None

    cfg = SPMConfig(
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        corpus=args.corpus,
        model_type=args.model_type,
        byte_fallback=(args.byte_fallback == "true"),
        normalization_rule_name=args.normalization_rule_name,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=(args.shuffle_input_sentence == "true"),
        character_coverage=args.character_coverage,
        hard_vocab_limit=(args.hard_vocab_limit == "true"),
        user_defined_symbols=uds,
    )

    # Build lexicon (anchors + neutrals)
    anchors = read_jsonl_texts(args.anchors, key="text")
    neutrals = read_jsonl_texts(args.neutrals, key="text")
    lex = extract_hazard_lexicon(
        anchors=anchors, neutrals=neutrals,
        min_stem_len=args.min_stem_len, max_hits_per_stem=1000
    )

    # Prior strategy
    prior = HazardAwarePrior(
        lex=lex,
        lambda_conf=args.lambda_conf,
        lambda_overlap=args.lambda_overlap,
        hazard_boost=args.hazard_boost,
        base_score=args.base_score,
        normalize_counts=True,
        count_norm=max(args.count_norm, 1.0),
    )

    # Candidate filter
    if args.drop_risky_substrings:
        cand_filter: CandidateFilter = DropRiskySubstrings(
            lex=lex, min_len=args.min_substring_len, drop_if_in=True
        )
    else:
        cand_filter = KeepAllFilter()

    # Extra seed pieces (optional)
    extra_seed: List[str] | None = None
    if args.extra_seed and os.path.exists(args.extra_seed):
        with open(args.extra_seed, "r", encoding="utf-8") as f:
            extra_seed = [ln.strip() for ln in f if ln.strip()]

    # Train
    trainer = SPMHazardTrainer(cfg=cfg, prior=prior, candidate_filter=cand_filter)
    model_path, vocab_path, train_log = trainer.train(
        anchors_path=args.anchors,
        neutrals_path=args.neutrals,
        min_stem_len=args.min_stem_len,
        extra_seed=extra_seed,
    )

    print(f"[done] wrote SPM model: {model_path}")
    print(f"[done] wrote SPM vocab: {vocab_path}")
    print(f"[note] hazard-aware seed written to: {cfg.model_prefix}_seed.txt")

    # Tiny JSON log
    log_path = f"{cfg.model_prefix}_hazard_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)
    print(f"[log] wrote hazard-aware SPM summary to: {log_path}")

    # Optional: export to a HF fast tokenizer folder with declared special tokens
    if args.export_hf_dir:
        os.makedirs(args.export_hf_dir, exist_ok=True)

        # Map hf_target to a sensible base model (used only to copy special tokens)
        target_to_base = {
            "llama3":   "meta-llama/Meta-Llama-3-8B",
            "mistral7b": "mistralai/Mistral-7B-v0.1",
            "qwen2_7b": "Qwen/Qwen2-7B"
        }
        base_model = target_to_base.get(args.hf_target, None)

        # Let users override via explicit flag (add this flag in build_cli below if needed)
        if getattr(args, "hf_base_model", None):
            base_model = args.hf_base_model

        # Use the dedicated exporter (writes tokenizer.json + tokenizer_config.json)
        try:
            from tools.tokenizer_export import spm_to_hf
            out_dir = spm_to_hf(model_path, args.export_hf_dir, base_model=base_model)
        except Exception as e:
            raise RuntimeError(
                f"[export] tokenizer_export failed: {type(e).__name__}: {e}"
            )

        # Optionally add/override special tokens and resave
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(out_dir)

        special_kwargs = {}
        if args.bos_token:
            special_kwargs["bos_token"] = args.bos_token
        if args.eos_token:
            special_kwargs["eos_token"] = args.eos_token
        if args.pad_token:
            special_kwargs["pad_token"] = args.pad_token

        extras = []
        if args.addl_special:
            extras = [s for s in args.addl_special.split(",") if s.strip()]
        if extras:
            special_kwargs["additional_special_tokens"] = extras

        if special_kwargs:
            tok.add_special_tokens(special_kwargs)
            tok.save_pretrained(out_dir)

        print(f"[export] Wrote HF tokenizer to: {out_dir}  (base_model={base_model})")


if __name__ == "__main__":
    main()
