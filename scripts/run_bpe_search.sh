#!/usr/bin/env bash
set -euo pipefail
BASE_TOK=${1:-"EleutherAI/pythia-410m"}
MODEL_ID=${2:-$BASE_TOK}
PROBE=${3:-"probes/v_layer.pt"}   # or .npy if you saved that way
python -m tokenizers.bpe_search \
  --base_tokenizer "$BASE_TOK" \
  --model_name "$MODEL_ID" \
  --anchors data/anchors.jsonl \
  --neutrals data/neutral_lookalikes.jsonl \
  --u_dev_dataset segyges/OpenWebText2 \
  --u_dev_size 20000 \
  --probe "$PROBE" \
  --rounds 5 \
  --prune_k 30 \
  --min_benign_hits 5 \
  --alpha 0.7 \
  --beta 0.1 \
  --warmup_steps 150 \
  --out tokenizers/bpe_searched
