setup:
	pip install -r requirements.txt

probe:
	bash scripts/run_probe.sh

train:
	bash scripts/run_lora_drift.sh

bpe_search:
	bash scripts/run_bpe_search.sh

eval:
	bash scripts/run_eval_all.sh
