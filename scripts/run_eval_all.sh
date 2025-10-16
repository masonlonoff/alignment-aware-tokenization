#!/usr/bin/env bash
set -euo pipefail
python -m eval.eval_perplexity --config configs/pythia410m.yml

python -m eval.eval_drift --config configs/pythia410m.yml --probe probes/v_layer.pt.npy

python -m eval.eval_jailbreak
  --config configs/llm_eval.yml \
  --probe probes/v_layer.pt.npy \
  --model_name adapters/pythia410m-bpe-searched-remap \
  --max_attack 1000 \
  --max_benign 1500
  
python -m eval.seg_stability \
  --tokenizer tokenizers/bpe_searched.json \
  --texts data/eval/benign_1500.jsonl \
  --key text \
  --max_texts 400 \
  --ops insert delete swap \
  --samples_per_op 3 \
  --bootstrap 800 \
  --seed 9172

