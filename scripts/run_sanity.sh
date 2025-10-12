set -euo pipefail
python -m tokenizers.spm_sanity \
  --model tokenizers/spm_hazard.model \
  --vocab tokenizers/spm_hazard.vocab \
  --anchors data/anchors.jsonl \
  --neutrals data/neutral_lookalikes.jsonl \
  --neutrals_sample 2000

# Pythia-1.4B (BPE)
# make setup
# bash scripts/run_lora_drift_pythia1_4b.sh

# # LLaMA-3-8B (SPM + priors before training)
# python -m tokenizers.spm_priors --corpus data/unlabeled/u_train.jsonl \
#   --out tokenizers/spm_hazard.model --vocab_size 50000 --pin ""  # adjust args
# python -m tokenizers.spm_sanity --model tokenizers/spm_hazard.model \
#   --vocab tokenizers/spm_hazard.vocab --anchors data/anchors/anchors_500.jsonl \
#   --neutrals data/neutrals/neutrals_1000.jsonl
# bash scripts/run_lora_drift_llama3_8b.sh
