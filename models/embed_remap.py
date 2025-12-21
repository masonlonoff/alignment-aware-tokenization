#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Embedding remap when swapping tokenizers.

Goal
-----
When you edit BPE merges or train a new SPM tokenizer, token strings/IDs can
change. To avoid a perplexity spike, initialize the *new* embedding matrix from
the *old* one by:
  (i) exact string copy if the piece exists in the old vocab;
  (ii) otherwise, decompose the piece with the *old* tokenizer and average the
       corresponding old embeddings (subpiece-sum/mean);
  (iii) optional fallback to nearest neighbor by cosine (string-sim is cheap).

If the LM ties input/output embeddings (common in causal LMs), we update both
`model.get_input_embeddings()` and `lm_head.weight` (if tied).

Design
------
- EmbeddingRemapper: orchestrates the swap; pure-PyTorch, no training step.
- Strategy-ish helpers for composition and neighbor fallback.

Usage
-----
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.embed_remap import EmbeddingRemapper

old_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", use_fast=True)
new_tok = AutoTokenizer.from_pretrained("tokenizers/bpe_searched", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m", dtype=torch.bfloat16).to("cuda")

remapper = EmbeddingRemapper()
model = remapper.remap(model, old_tok, new_tok, average_pool=True, update_lm_head=True)
model.save_pretrained("adapters/pythia410m-bpe-searched-remap")
new_tok.save_pretrained("adapters/pythia410m-bpe-searched-remap")
"""

from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, PreTrainedModel


def _build_piece_to_id(tok: PreTrainedTokenizerBase) -> Dict[str, int]:
    """Map token string -> id for fast lookup (handles fast tokenizers too)."""
    # For fast tokenizers, `get_vocab()` returns {piece: id}
    vocab = tok.get_vocab()
    # Make sure ties are consistent (prefer the smallest id for duplicates)
    piece_to_id: Dict[str, int] = {}
    for p, ix in vocab.items():
        if p not in piece_to_id or ix < piece_to_id[p]:
            piece_to_id[p] = ix
    return piece_to_id


def _tokenize_old(tok_old: PreTrainedTokenizerBase, piece: str) -> List[int]:
    """Tokenize the *string form of a new piece* under the old tokenizer."""
    # Disable specials so we only get content pieces
    return tok_old(piece, add_special_tokens=False)["input_ids"]


class EmbeddingRemapper:
    """
    Remap embeddings from (old_tok, model) → (new_tok) without retraining.

    Steps:
      1) Build old string→id map and fetch old embedding matrix.
      2) For each new piece:
         - exact copy if present in old vocab (fast path),
         - else average old subpiece embeddings from old tokenize(piece),
         - else (rare) cosine-nearest neighbor among old embeddings.
      3) Write remapped vectors into the model's input embedding table.
      4) If `update_lm_head=True` and `lm_head.weight` is tied, sync it too.

    Args:
      nn_fallback_topk: if >0, tries cosine-NN fallback when composition is empty.
      device: torch device to run tiny ops on (defaults to model device).
    """

    def __init__(self, nn_fallback_topk: int = 16, device: Optional[str] = None):
        self.nn_fallback_topk = nn_fallback_topk
        self.device = device

    @staticmethod
    def _maybe_get_lm_head(model: PreTrainedModel) -> Optional[torch.nn.Parameter]:
        """Return lm_head.weight if present (causal LMs), else None."""
        head = getattr(model, "lm_head", None)
        if head is None:
            return None
        return getattr(head, "weight", None)

    def _nearest_neighbor(
        self,
        target_text: str,
        old_tok: PreTrainedTokenizerBase,
        old_emb: torch.Tensor,
        piece_to_id_old: Dict[str, int],
        topk: int = 16,
    ) -> Optional[torch.Tensor]:
        """
        Fallback: embed `target_text` via old tokenizer and mean; if empty,
        return cosine-NN vector among *string-equal* approximations.
        """
        ids = _tokenize_old(old_tok, target_text)
        if ids:
            return old_emb[torch.tensor(ids, device=old_emb.device)].mean(0)

        # No decomposition (very rare). Try a crude character-NN: pick old piece
        # with highest Jaccard on character bigrams; then return its vector.
        def char_bigrams(s: str) -> set:
            s = s.replace(" ", "")
            return set(zip(s, s[1:])) if len(s) >= 2 else set()

        tg = char_bigrams(target_text)
        best, best_sim = None, -1.0
        for p, ix in piece_to_id_old.items():
            bg = char_bigrams(p)
            inter = len(tg & bg)
            union = len(tg | bg) or 1
            jac = inter / union
            if jac > best_sim:
                best_sim, best = p, ix
        if best is not None:
            return old_emb[best]
        return None

    def remap(
        self,
        model: PreTrainedModel,
        old_tok: PreTrainedTokenizerBase,
        new_tok: PreTrainedTokenizerBase,
        average_pool: bool = True,
        update_lm_head: bool = True,
    ) -> PreTrainedModel:
        """
        Build a new embedding matrix in-place for `model` that matches `new_tok`.

        Args:
          model:  Pretrained LM whose embeddings will be remapped in-place.
          old_tok: Tokenizer the model was trained with (source of embeddings).
          new_tok: Target tokenizer (strings define target rows to initialize).
          average_pool: If True, mean-pool old subpieces for unseen tokens.
          update_lm_head: If True, update lm_head.weight if present/tied.

        Returns:
          The same model instance with updated embeddings (and lm_head if tied).
        """
        device = self.device or next(model.parameters()).device
        dtype = model.get_input_embeddings().weight.dtype

        # Old embedding matrix
        old_emb = model.get_input_embeddings().weight.detach()  # [V_old, H]
        V_old, H = old_emb.shape
        piece_to_id_old = _build_piece_to_id(old_tok)

        # 🔧 IMPORTANT: use *len(tokenizer)*, not vocab_size, so we include added tokens.
        V_new = len(new_tok)
        new_weight = torch.empty((V_new, H), dtype=dtype, device=device)

        # Precompute normalized matrix for cheap cosine-NN if needed
        old_norm = F.normalize(old_emb, dim=1) if self.nn_fallback_topk > 0 else None

        # Iterate new vocab in id order to fill rows
        # For fast tokenizers, `convert_ids_to_tokens(i)` gets the string
        for i in range(V_new):
            piece = new_tok.convert_ids_to_tokens(i)
            if piece is None:
                piece = ""

            # 1) exact copy
            if piece in piece_to_id_old:
                new_weight[i] = old_emb[piece_to_id_old[piece]].to(device=device, dtype=dtype)
                continue

            # 2) composition with old tokenizer
            vec = None
            if average_pool:
                ids = _tokenize_old(old_tok, piece)
                if len(ids) > 0:
                    vec = old_emb[
                        torch.tensor(ids, device=old_emb.device)
                    ].mean(0).to(device=device, dtype=dtype)

            # 3) fallback NN by cosine on old pieces
            if vec is None and old_norm is not None:
                ids = _tokenize_old(old_tok, piece)
                if ids:
                    target = F.normalize(
                        old_emb[torch.tensor(ids, device=old_emb.device)].mean(0).unsqueeze(0),
                        dim=1,
                    )
                    sims = (old_norm @ target.T).squeeze(1)  # [V_old]
                    topk = min(self.nn_fallback_topk, V_old)
                    nn_ix = torch.topk(sims, k=topk, dim=0).indices
                    vec = old_emb[nn_ix].mean(0).to(device=device, dtype=dtype)
                else:
                    # Last resort: zero init (rare)
                    vec = torch.zeros(H, dtype=dtype, device=device)

            new_weight[i] = vec

        # Write back
        input_emb = model.get_input_embeddings()
        old_n, d_old = input_emb.weight.shape
        new_n, d_new = new_weight.shape

        if d_old != d_new:
            raise ValueError(
                f"Hidden size mismatch between model emb ({d_old}) and new_weight ({d_new})."
            )

        # If vocab sizes differ, resize model embeddings to new_n
        if new_n != old_n:
            # this will resize both input and output embeddings if tie_word_embeddings=True
            model.resize_token_embeddings(new_n)
            input_emb = model.get_input_embeddings()

        # Copy new weights into resized embedding matrix
        input_emb.weight.data.copy_(
            new_weight.to(input_emb.weight.device, dtype=input_emb.weight.dtype)
        )

        # Optionally tie / copy into LM head as well
        if update_lm_head:
            out_emb = model.get_output_embeddings()
            if out_emb is not None and out_emb.weight.shape == new_weight.shape:
                out_emb.weight.data.copy_(
                    new_weight.to(out_emb.weight.device, dtype=out_emb.weight.dtype)
                )

            # If tied lm_head exists, mirror the weights
            lm_head_w = self._maybe_get_lm_head(model)
            if lm_head_w is not None and lm_head_w.shape == new_weight.shape:
                lm_head_w.data[:] = new_weight.to(
                    lm_head_w.device, dtype=lm_head_w.dtype
                )

        return model
