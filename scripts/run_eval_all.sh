#!/usr/bin/env bash
set -euo pipefail
python -m eval.eval_perplexity --config configs/pythia410m.yml

python -m eval.eval_drift --config configs/pythia410m.yml --probe probes/v_layer.pt.npy

python -m eval.eval_jailbreak \
  --config configs/llm_eval.yml \
  --probe probes/v_layer.pt.npy \
  --model_name mistralai/Mistral-7B-v0.1 \ # make sure to change to your adapter path
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
  --tokenizer tokenizers/bpe_searched \
  --texts data/eval/benign_1500.jsonl \
  --key text \
  --max_texts 400 \
  --ops insert delete swap \
  --samples_per_op 3 \
  --bootstrap 800 \
  --seed 9172

python -m eval.label_efficiency --config configs/pythia410m.yml --budgets 50 100 300 --trials 10 --check_overlap --dedup
python -m eval.label_efficiency --config configs/pythia1_4b.yml --budgets 50 100 300 --trials 10 --check_overlap --dedup
python -m eval.label_efficiency --config configs/mistral7b.yml   --budgets 50 100 300 --trials 10 --check_overlap --dedup
python -m eval.label_efficiency --config configs/llama3_8b.yml --budgets 50 100 300 --trials 10 --check_overlap --dedup
python -m eval.label_efficiency --config configs/qwen2_7b.yml --budgets 50 100 300 --trials 10 --check_overlap --dedup
