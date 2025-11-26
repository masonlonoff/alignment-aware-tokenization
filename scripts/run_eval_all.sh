#!/usr/bin/env bash
set -euo pipefail
python -m eval.eval_perplexity --config configs/pythia410m.yml

python -m eval.eval_drift --config configs/pythia410m.yml --probe probes/v_layer.pt.npy

python -m eval.eval_jailbreak \
  --config configs/llm_eval.yml \
  --probe probes/v_layer.pt.npy \
  --model_name adapters/pythia410m-bpe-searched-remap \ # make sure to change to your adapter path
  --atk_n 1000 \
  --benign_n 1500 \
  --calib_n 256 \
  --score_mode resp_only \
  --benign_fpr 0.01 \
  --greedy \
  --max_new_tokens 32 \
  --gen_bs 16 \
  --batch_size 32 \
  --refusal_window 200 \
  --dedup_attacks
  
python -m eval.seg_stability \
  --tokenizer tokenizers/bpe_searched.json \
  --texts data/eval/benign_1500.jsonl \
  --key text \
  --max_texts 400 \
  --ops insert delete swap \
  --samples_per_op 3 \
  --bootstrap 800 \
  --seed 9172

