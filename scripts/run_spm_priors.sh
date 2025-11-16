# #!/usr/bin/env bash
# # Run hazard-aware SPM priors → export HF tokenizer.json in one go.
# # Usage:
# set -euo pipefail

# # these special tokens can be override/augment explicitly
# #   --bos_token "<s>" --eos_token "</s>" --pad_token "<pad>" \
# #   --addl_special "<|im_start|>,<|im_end|>"


# # LLaMA-3-8B preset
# python -m tokenizers.spm_priors \
#   --hf_target llama3 \
#   --corpus data/unlabeled/u_train.jsonl \
#   --anchors data/anchors/anchors_500.jsonl \
#   --neutrals data/neutrals/neutrals_1000.jsonl \
#   --model_prefix tokenizers/spm_hazard/spm_llama3_hazard \
#   --vocab_size 50000 \
#   --lambda_conf 0.3 \
#   --lambda_overlap 0.5 \
#   --hazard_boost 5.0 \
#   --normalization_rule_name identity \
#   --export_hf_dir tokenizers/spm_hazard/hf_spm_llama3_hazard \
#   --addl_special "<s>,</s>"

# # Mistral-7B
# python -m tokenizers.spm_priors \
#   --hf_target mistral7b \
#   --corpus data/unlabeled/u_train.jsonl \
#   --anchors data/anchors/anchors_500.jsonl \
#   --neutrals data/neutrals/neutrals_1000.jsonl \
#   --model_prefix tokenizers/spm_hazard/spm_mistral7b_hazard \
#   --vocab_size 50000 \
#   --lambda_conf 0.3 --lambda_overlap 0.5 --hazard_boost 5.0 \
#   --normalization_rule_name identity \
#   --export_hf_dir tokenizers/spm_hazard/hf_spm_mistral7b_hazard \
#   --addl_special "<s>,</s>"

# # Qwen2-7B
# python -m tokenizers.spm_priors \
#   --hf_target qwen2_7b \
#   --corpus data/unlabeled/u_train.jsonl \
#   --anchors data/anchors/anchors_500.jsonl \
#   --neutrals data/neutrals/neutrals_1000.jsonl \
#   --model_prefix tokenizers/spm_hazard/spm_qwen2-7b_hazard \
#   --vocab_size 50000 \
#   --lambda_conf 0.3 --lambda_overlap 0.5 --hazard_boost 5.0 \
#   --normalization_rule_name identity \
#   --export_hf_dir tokenizers/spm_hazard/hf_spm_qwen2-7b_hazard \
#   --addl_special "<s>,</s>"

#!/usr/bin/env bash
set -euo pipefail

# Choose a preset by model family (affects only naming/user symbols)
FAMILY="${1:-mistral-7b}"   # options: llama3-8b | mistral-7b | qwen2-7b
OUT_DIR="tokenizers/spm_hazard"
DATA_CORPUS="data/unlabeled/u_train.jsonl"
ANCHORS="data/anchors/anchors_500.jsonl"
NEUTRALS="data/neutrals/neutrals_1000.jsonl"

mkdir -p "$OUT_DIR"
PREFIX="${OUT_DIR}/spm_${FAMILY}_hazard"
EXPORT_DIR="${OUT_DIR}/spm_${FAMILY}_hazard_hf"

python -m tokenizers.spm_priors \
  --corpus "$DATA_CORPUS" \
  --anchors "$ANCHORS" \
  --neutrals "$NEUTRALS" \
  --model_prefix "$PREFIX" \
  --vocab_size 50000 \
  --lambda_conf 0.3 \
  --lambda_overlap 0.25 \
  --hazard_boost 5.0 \
  --min_stem_len 3 \
  --input_sentence_size 200000 \
  --byte_fallback true \
  --hf_target "$FAMILY" \
  --export_hf_dir "$EXPORT_DIR" \
  --user_defined_symbols "<s>,</s>,<pad>"  # safe across families
