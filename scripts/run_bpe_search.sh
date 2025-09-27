#!/usr/bin/env bash
set -euo pipefail
BASE_TOK=${1:-"EleutherAI/pythia-410m"}
python -m tokenizers.bpe_search --base_tokenizer $BASE_TOK \
  --anchors data/anchors.jsonl --neutrals data/neutral_lookalikes.jsonl \
  --u_dev_size 20000 --rounds 5 --out tokenizers/bpe_searched.json
