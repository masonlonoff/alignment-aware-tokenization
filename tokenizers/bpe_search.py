# tokenizers/bpe_search.py
"""
Drift-aware bi-level BPE merge search.

This module implements a practical search procedure that edits the BPE merge list
to reduce "subword spillover" for safety-sensitive stems while preserving language
model quality. It follows the plan in the proposal:

  1) Load a baseline fast BPE tokenizer and extract its merges from tokenizer.json.
  2) Detect "risky" merges whose concatenation equals a hazard stem that frequently
     appears in neutral texts (proxy for spillover risk).
  3) Iteratively prune a small batch of risky merges (propose), do a quick LoRA
     warmup to adapt the LM, then score each candidate tokenizer using:

        J = ppl/ppl0 + α * drift/drift0 + β * tpc/tpc0

     where:
       - ppl  = perplexity on held-out U_dev
       - drift = neutral-context concept logit (lower is better)
       - tpc  = tokens per character (length/latency proxy)
       - *_0  = baseline normalization constants
  4) Accept edits only if J improves; output a new HF tokenizer directory with
     edited `tokenizer.json` (same vocab + specials; merges changed).

Design patterns:
  - Strategy:   ScoreStrategy, CandidateGenerator
  - Template:   BPESearch.search() is the template method driving rounds
  - Encapsulation: TokenizerEditor (I/O), WarmupAdapter (LoRA warmup)

Typical usage (CLI):
  python -m tokenizers.bpe_search --base_tokenizer EleutherAI/pythia-410m \
    --model_name EleutherAI/pythia-410m \
    --anchors data/anchors.jsonl --neutrals data/neutral_lookalikes.jsonl \
    --u_dev_dataset segyges/OpenWebText2 --u_dev_size 20000 \
    --probe probes/v_layer.pt --rounds 5 --prune_k 30 \
    --out tokenizers/bpe_searched
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from eval.eval_perplexity import perplexity as eval_ppl
from eval.eval_drift import drift_score as eval_drift


# ----------------------------
# Utilities
# ----------------------------

def set_seeds(seed: int = 9172):
    """
    Set Python, NumPy, and PyTorch RNG seeds for determinism (best-effort).

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_jsonl_texts(path: str, key: str = "text", limit: int | None = None) -> List[str]:
    """
    Read newline-delimited JSON with a given `key` (default: "text").

    Args:
        path: File path to JSONL.
        key:  Field name to pull the text from.
        limit: Optional maximum number of lines to read.

    Returns:
        List of non-empty, stripped strings.
    """
    out = []
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


def sample_u_dev(dataset: str, n: int, seed: int) -> List[str]:
    """
    Sample a held-out slice from a HuggingFace dataset for ppl/tpc evaluation.

    Args:
        dataset: HF dataset name (split must support "train").
        n:       Sample size (capped by dataset length).
        seed:    Shuffle seed.

    Returns:
        A list of `text` fields.
    """
    ds = load_dataset(dataset, split="train").shuffle(seed=seed)
    n = min(n, len(ds))
    return [x.get("text", "") for x in ds.select(range(n))]


def tokens_per_char(tok: PreTrainedTokenizerFast, texts: List[str]) -> float:
    """
    Compute tokens-per-character over a corpus (lower is more compressive).

    Args:
        tok:   HF fast tokenizer.
        texts: List of strings.

    Returns:
        Mean tokens per character (float).
    """
    total_toks = total_chars = 0
    for t in texts:
        if not t:
            continue
        ids = tok(t, add_special_tokens=False)["input_ids"]
        total_toks += len(ids)
        total_chars += len(t)
    return (total_toks / max(total_chars, 1)) if total_chars else math.inf


def extract_hazard_stems(anchors: List[str], min_len: int = 3) -> List[str]:
    """
    Extract simple alphanumeric "stems" from hazard anchors.

    Rationale:
        We use a coarse stem proxy by lowercasing, removing non-alphanumerics,
        and keeping units of length >= min_len. This is sufficient to identify
        merges that exactly concatenate into these stems (e.g., "bo"+"mb"->"bomb").

    Args:
        anchors: Hazard anchor texts.
        min_len: Minimum stem length to keep.

    Returns:
        Sorted unique stems.
    """
    stems = set()
    for t in anchors:
        for w in t.lower().split():
            w = "".join(ch for ch in w if ch.isalnum())
            if len(w) >= min_len:
                stems.add(w)
    return sorted(stems)


def benign_counts_by_stem(neutrals: List[str], stems: List[str], max_hits_per_stem: int = 200) -> Dict[str, int]:
    """
    Count how often each hazard stem appears as a substring inside neutral texts.

    Args:
        neutrals: Neutral look-alike sentences.
        stems:    Hazard stems to test.
        max_hits_per_stem: Early-stop threshold per stem for speed.

    Returns:
        Dict stem -> count (number of neutral sentences containing the stem).
    """
    counts = {s: 0 for s in stems}
    for s in stems:
        s_low = s.lower()
        hits = 0
        for text in neutrals:
            if s_low in text.lower():
                hits += 1
                if hits >= max_hits_per_stem:
                    break
        counts[s] = hits
    return counts


# ----------------------------
# Tokenizer editor
# ----------------------------

class TokenizerEditor:
    """
    Read/modify the merges array inside a fast BPE tokenizer.json and
    build an edited HF fast tokenizer directory.

    Note:
        We preserve vocab/special tokens and only change the 'merges' list.

    Attributes:
        baseline_tok_dir: Temp directory with a saved fast tokenizer.
        baseline_merges:  Original merges list.
    """

    def __init__(self, baseline_tok_dir: Path):
        self.baseline_tok_dir = baseline_tok_dir
        self.tokjson_path = baseline_tok_dir / "tokenizer.json"
        if not self.tokjson_path.exists():
            raise FileNotFoundError("tokenizer.json not found; need a fast BPE tokenizer.")

        obj = json.loads(self.tokjson_path.read_text(encoding="utf-8"))
        model = obj.get("model", {})
        if model.get("type", "").lower() != "bpe":
            raise ValueError("Expected a BPE fast tokenizer (model.type != 'BPE').")
        if "merges" not in model:
            raise ValueError("No 'merges' array in tokenizer.json.")

        self.baseline_merges: List[str] = list(model["merges"])

    def write_edited(self, merges: List[str], out_dir: Path) -> PreTrainedTokenizerFast:
        """
        Write a new tokenizer dir with edited merges; return the loaded tokenizer.

        Args:
            merges:  New merges list to store in tokenizer.json.
            out_dir: Output directory to create.

        Returns:
            A loaded PreTrainedTokenizerFast pointing at the edited tokenizer.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        # Copy all files except tokenizer.json so special tokens/config are preserved.
        for f in self.baseline_tok_dir.glob("*"):
            if f.name == "tokenizer.json":
                continue
            shutil.copy2(f, out_dir / f.name)

        # Replace merges inside tokenizer.json
        obj = json.loads(self.tokjson_path.read_text(encoding="utf-8"))
        obj["model"]["merges"] = merges
        (out_dir / "tokenizer.json").write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

        # Load fast tokenizer from edited file
        tok = PreTrainedTokenizerFast(tokenizer_file=str(out_dir / "tokenizer.json"))
        # Try to propagate special tokens
        try:
            base = AutoTokenizer.from_pretrained(str(self.baseline_tok_dir))
            tok.pad_token = base.pad_token
            tok.eos_token = base.eos_token
            tok.bos_token = getattr(base, "bos_token", None)
        except Exception:
            pass
        return tok


# ----------------------------
# Candidate generation (Strategy)
# ----------------------------

class CandidateGenerator:
    """
    Strategy base for proposing merge edits.

    Subclasses should implement `propose(merges)` and return indices of
    merges to prune for the current round.
    """

    def propose(self, merges: List[str]) -> List[int]:
        raise NotImplementedError


class RiskyMergePruner(CandidateGenerator):
    """
    Propose pruning merges that concatenate exactly into hazard stems
    that frequently appear in neutral texts.

    Heuristic:
        For each merge "a b", compute cat = "ab". If cat is a hazard stem
        and that stem appears >= min_benign_hits times as a substring in
        neutral sentences, consider the merge risky and eligible for pruning.

    Args:
        stems:           Hazard stems.
        benign_counts:   Map stem -> count of neutral sentences containing it.
        min_benign_hits: Minimum neutral-occurrence threshold to mark risky.
        prune_k:         Max risky merges to prune in a round.
    """

    def __init__(self, stems: List[str], benign_counts: Dict[str, int], min_benign_hits: int, prune_k: int):
        self.stems = set(stems)
        self.counts = dict(benign_counts)
        self.min_hits = int(min_benign_hits)
        self.k = int(prune_k)

    def risky_indices(self, merges: List[str]) -> List[int]:
        """
        Compute the indices in `merges` that are considered risky.

        Args:
            merges: Current merges list.

        Returns:
            List of integer indices into `merges`.
        """
        risky = []
        for i, m in enumerate(merges):
            parts = m.split()
            if len(parts) != 2:
                continue
            cat = "".join(parts)
            if (cat in self.stems) and (self.counts.get(cat, 0) >= self.min_hits):
                risky.append(i)
        return risky

    def propose(self, merges: List[str]) -> List[int]:
        """
        Choose up to `prune_k` risky merge indices for pruning.

        Args:
            merges: Current merges list.

        Returns:
            Sorted list of indices to remove (can be empty).
        """
        risky = self.risky_indices(merges)
        if not risky:
            return []
        k = min(self.k, len(risky))
        return sorted(random.sample(risky, k))


# ----------------------------
# Warmup adapter (LoRA)
# ----------------------------

class WarmupAdapter:
    """
    Perform a short LoRA warmup to let the LM adapt to the new tokenizer
    before computing ppl/drift. This stabilizes scoring after merge edits.

    Args:
        model_name:   HF model id to load for scoring (same family as base).
        steps:        Number of single-text steps (each a tiny "batch").
        lr:           AdamW learning rate for warmup.
        lora_r:       LoRA rank.
        lora_alpha:   LoRA alpha.
        lora_dropout: LoRA dropout.
        device:       Torch device (default "cuda").
    """

    def __init__(self, model_name: str, steps: int = 150, lr: float = 2e-4,
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05, device: str = "cuda"):
        self.model_name = model_name
        self.steps = steps
        self.lr = lr
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.device = device

    def run(self, tok: PreTrainedTokenizerFast, texts: List[str]) -> AutoModelForCausalLM:
        """
        Run LoRA warmup on a small slice of `texts`.

        Args:
            tok:   Edited tokenizer to condition model inputs.
            texts: List of strings to run small adaptation steps.

        Returns:
            A PEFT-wrapped AutoModelForCausalLM in eval mode.
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16).to(self.device)
        lcfg = LoraConfig(
            r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout,
            target_modules=["Wqkv", "out_proj", "fc_in", "fc_out"]
        )
        model = get_peft_model(model, lcfg)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr)

        it = 0
        for t in texts:
            if it >= self.steps:
                break
            if not t:
                continue
            batch = tok(t, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            out = model(**batch, labels=batch["input_ids"])
            out.loss.backward()
            opt.step()
            model.zero_grad(set_to_none=True)
            it += 1
        model.eval()
        return model


# ----------------------------
# Scoring (Strategy)
# ----------------------------

@dataclass
class ScoreResult:
    """
    Container for scoring outputs.

    Attributes:
        J:     Joint objective value.
        ppl:   Perplexity on U_dev.
        drift: Neutral drift score (mean concept logit).
        tpc:   Tokens per character on U_dev.
    """
    J: float
    ppl: float
    drift: float
    tpc: float


class ScoreStrategy:
    """
    Strategy base for candidate tokenizer scoring.
    """

    def score(self, tok: PreTrainedTokenizerFast) -> ScoreResult:
        raise NotImplementedError


class JointScore(ScoreStrategy):
    """
    Joint scoring with normalization:

        J = ppl/ppl0 + alpha * (drift/drift0) + beta * (tpc/tpc0)

    Each candidate is briefly LoRA-warmed-up to reduce adaptation artifacts.

    Args:
        model_name:   Model id to load for warmup + scoring.
        u_dev_texts:  Held-out texts for ppl/tpc.
        neutrals:     Neutral look-alikes for drift.
        v_path:       Path to probe vector (NumPy .npy or same saved as .pt.npy).
        alpha:        Weight for drift term.
        beta:         Weight for tpc term.
        ppl0:         Baseline ppl (normalizer).
        drift0:       Baseline drift (normalizer).
        tpc0:         Baseline tpc (normalizer).
        drift_layer:  Hidden layer index for concept projection.
        warmup_steps: LoRA warmup steps per candidate.
    """

    def __init__(self, model_name: str, u_dev_texts: List[str], neutrals: List[str], v_path: str,
                 alpha: float, beta: float, ppl0: float, drift0: float, tpc0: float,
                 drift_layer: int = 10, warmup_steps: int = 150):
        self.model_name = model_name
        self.u_dev_texts = u_dev_texts
        self.neutrals = neutrals
        self.alpha = alpha
        self.beta = beta
        self.ppl0 = ppl0
        self.drift0 = drift0
        self.tpc0 = tpc0
        self.drift_layer = drift_layer
        self.v = self._load_v(v_path)
        self._warmup = WarmupAdapter(model_name, steps=warmup_steps)

    @staticmethod
    def _load_v(path: str) -> np.ndarray:
        """
        Load concept vector `v` saved as .npy (or .pt.npy).

        Args:
            path: File path without/with .npy extension.

        Returns:
            NumPy array (1, H) or (H,) normalized vector.
        """
        if path.endswith(".npy"):
            return np.load(path, allow_pickle=True)
        if os.path.exists(path + ".npy"):
            return np.load(path + ".npy", allow_pickle=True)
        return np.load(path, allow_pickle=True)

    def score(self, tok: PreTrainedTokenizerFast) -> ScoreResult:
        """
        Score a candidate tokenizer by:
          - quick LoRA warmup,
          - computing ppl, tpc, and drift,
          - returning the normalized joint objective J.

        Args:
            tok: Candidate tokenizer to evaluate.

        Returns:
            ScoreResult(J, ppl, drift, tpc).
        """
        model = self._warmup.run(tok, self.u_dev_texts[: self._warmup.steps])

        ppl = eval_ppl(model, tok, self.u_dev_texts)
        tpc = tokens_per_char(tok, self.u_dev_texts)
        drift = eval_drift(model, tok, self.neutrals, self.v, layer=self.drift_layer)

        J = (ppl / self.ppl0) + self.alpha * (drift / self.drift0) + self.beta * (tpc / self.tpc0)
        return ScoreResult(float(J), float(ppl), float(drift), float(tpc))


# ----------------------------
# Search controller (Template Method)
# ----------------------------

class BPESearch:
    """
    Controller implementing the round-based prune→warmup→score→accept loop.

    Args:
        base_tokenizer_id: HF id for the baseline fast BPE tokenizer.
        model_name:        HF id for the LM used in scoring.
        anchors:           Hazard anchors (used to extract stems).
        neutrals:          Neutral look-alikes (drift evaluation).
        u_dev_texts:       Held-out texts (ppl/tpc evaluation).
        score:             ScoreStrategy instance (e.g., JointScore).
        candidate_gen:     CandidateGenerator instance (e.g., RiskyMergePruner).
        rounds:            Number of search rounds.
        seed:              RNG seed.
    """

    def __init__(self, base_tokenizer_id: str, model_name: str, anchors: List[str], neutrals: List[str],
                 u_dev_texts: List[str], score: ScoreStrategy, candidate_gen: CandidateGenerator,
                 rounds: int, seed: int):
        set_seeds(seed)
        self.base_tokenizer_id = base_tokenizer_id
        self.model_name = model_name
        self.anchors = anchors
        self.neutrals = neutrals
        self.u_dev_texts = u_dev_texts
        self.score = score
        self.candidate_gen = candidate_gen
        self.rounds = rounds

        # Prepare editable baseline tokenizer dir
        self._tmpdir = Path(tempfile.mkdtemp())
        self.base_tok = AutoTokenizer.from_pretrained(self.base_tokenizer_id, use_fast=True)
        if not isinstance(self.base_tok, PreTrainedTokenizerFast):
            raise ValueError("Need a fast BPE tokenizer.")
        self.base_tok.save_pretrained(self._tmpdir)
        self.editor = TokenizerEditor(self._tmpdir)

        self.best_merges = list(self.editor.baseline_merges)
        self.best_tok = self.base_tok
        self.best_score: ScoreResult | None = None

    def _tok_from_merges(self, merges: List[str]) -> PreTrainedTokenizerFast:
        """
        Build a temporary tokenizer from `merges` for evaluation.

        Args:
            merges: Candidate merges list.

        Returns:
            Loaded fast tokenizer instance for scoring.
        """
        tmp = Path(tempfile.mkdtemp())
        return self.editor.write_edited(merges, tmp)

    def initialize_baseline(self):
        """
        Score the baseline tokenizer in the same pipeline for comparable logs.
        """
        res = self.score.score(self.base_tok)
        self.best_score = res
        print(
            f"[round 0] J={res.J:.4f} | ppl={res.ppl:.3f} | drift={res.drift:.5f} | tpc={res.tpc:.5f}")

    def search(self) -> Tuple[List[str], ScoreResult]:
        """
        Run R rounds of candidate generation and accept edits that improve J.

        Returns:
            (best_merges, best_score) where best_score is a ScoreResult.
        """
        merges = list(self.best_merges)
        for r in range(1, self.rounds + 1):
            remove_idx = self.candidate_gen.propose(merges)
            if not remove_idx:
                print(f"[round {r}] no candidates to prune; stopping.")
                break

            # Apply prune
            keep = [True] * len(merges)
            for idx in remove_idx:
                if 0 <= idx < len(merges):
                    keep[idx] = False
            new_merges = [m for m, k in zip(merges, keep) if k]

            # Score candidate
            cand_tok = self._tok_from_merges(new_merges)
            cand_res = self.score.score(cand_tok)
            print(f"[round {r}] removed={len(remove_idx)} | J={cand_res.J:.4f} | "
                  f"ppl={cand_res.ppl:.3f} | drift={cand_res.drift:.5f} | tpc={cand_res.tpc:.5f}")

            # Accept/reject
            if (self.best_score is None) or (cand_res.J < self.best_score.J):
                print(
                    f"  -> accepted (J ↓ {self.best_score.J:.4f} → {cand_res.J:.4f})" if self.best_score else "  -> accepted")
                merges = new_merges
                self.best_merges = new_merges
                self.best_score = cand_res
            else:
                print("  -> rejected (no improvement)")
        return self.best_merges, self.best_score  # type: ignore


# ----------------------------
# CLI
# ----------------------------

def main():
    """
    Command-line entry point. See module docstring for a usage example.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_tokenizer", required=True,
                    help="HF id for fast BPE tokenizer, e.g., EleutherAI/pythia-410m")
    ap.add_argument("--model_name", default=None,
                    help="LM id used for warmup/scoring; defaults to base_tokenizer")
    ap.add_argument("--anchors", required=True, help="JSONL with {text: ...} hazard anchors")
    ap.add_argument("--neutrals", required=True, help="JSONL with {text: ...} neutral look-alikes")
    ap.add_argument("--u_dev_dataset", default="segyges/OpenWebText2",
                    help="HF dataset for ppl/tpc dev slice")
    ap.add_argument("--u_dev_size", type=int, default=20000, help="Sample size for dev slice")
    ap.add_argument("--probe", required=True,
                    help="Path to concept vector (npy or pt saved as npy)")
    ap.add_argument("--rounds", type=int, default=5, help="Search rounds")
    ap.add_argument("--prune_k", type=int, default=30, help="Max risky merges to prune per round")
    ap.add_argument("--min_benign_hits", type=int, default=5,
                    help="Min neutral occurrences to mark a stem risky")
    ap.add_argument("--alpha", type=float, default=0.7, help="Weight for drift term in J")
    ap.add_argument("--beta", type=float, default=0.1, help="Weight for tpc term in J")
    ap.add_argument("--drift_layer", type=int, default=10, help="Layer index for drift projection")
    ap.add_argument("--warmup_steps", type=int, default=150, help="LoRA warmup steps per candidate")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--max_h", type=int, default=1000, help="Max anchors to read")
    ap.add_argument("--max_n", type=int, default=2000, help="Max neutrals to read")
    ap.add_argument("--out", required=True, help="Output dir for edited tokenizer (HF format)")
    args = ap.parse_args()

    set_seeds(args.seed)
    model_name = args.model_name or args.base_tokenizer

    anchors = read_jsonl_texts(args.anchors, limit=args.max_h)
    neutrals = read_jsonl_texts(args.neutrals, limit=args.max_n)
    u_dev_texts = sample_u_dev(args.u_dev_dataset, args.u_dev_size, seed=args.seed)

    stems = extract_hazard_stems(anchors)
    counts = benign_counts_by_stem(neutrals, stems)

    # Normalization constants from the unedited baseline (no warmup).
    base_tok = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    if not isinstance(base_tok, PreTrainedTokenizerFast):
        raise ValueError("Need a fast BPE tokenizer.")
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda").eval()
    ppl0 = eval_ppl(base_model, base_tok, u_dev_texts)
    tpc0 = tokens_per_char(base_tok, u_dev_texts)
    v = np.load(args.probe, allow_pickle=True) if args.probe.endswith(".npy") else (
        np.load(args.probe + ".npy", allow_pickle=True) if os.path.exists(args.probe +
                                                                          ".npy") else np.load(args.probe, allow_pickle=True)
    )
    drift0 = eval_drift(base_model, base_tok, neutrals, v, layer=args.drift_layer)
    print(f"[baseline norms] ppl0={ppl0:.3f} | tpc0={tpc0:.5f} | drift0={drift0:.5f}")

    scorer = JointScore(model_name, u_dev_texts, neutrals, args.probe,
                        alpha=args.alpha, beta=args.beta,
                        ppl0=ppl0, drift0=drift0, tpc0=tpc0,
                        drift_layer=args.drift_layer, warmup_steps=args.warmup_steps)
    generator = RiskyMergePruner(
        stems, counts, min_benign_hits=args.min_benign_hits, prune_k=args.prune_k)

    search = BPESearch(args.base_tokenizer, model_name, anchors, neutrals, u_dev_texts,
                       score=scorer, candidate_gen=generator, rounds=args.rounds, seed=args.seed)
    search.initialize_baseline()
    best_merges, best_score = search.search()

    out_dir = Path(args.out)
    editor = search.editor
    editor.write_edited(best_merges, out_dir)
    print(f"[done] wrote searched tokenizer to: {out_dir}")
    print(f"[best] J={best_score.J:.4f} | ppl={best_score.ppl:.3f} | drift={best_score.drift:.5f} | tpc={best_score.tpc:.5f}")


if __name__ == "__main__":
    main()
