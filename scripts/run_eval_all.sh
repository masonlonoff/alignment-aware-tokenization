#!/usr/bin/env bash
set -euo pipefail
python -m eval.eval_perplexity --config configs/pythia410m.yml
python -m eval.eval_drift --config configs/pythia410m.yml --probe probes/v_layer.pt
python -m eval.eval_jailbreak --config configs/llm_eval.yml --probe probes/v_layer.pt
python -m eval.seg_stability --base_tokenizer tokenizers/bpe_searched.json
