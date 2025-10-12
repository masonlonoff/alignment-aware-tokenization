#!/usr/bin/env bash
set -euo pipefail
python -m models.probe --config configs/llama3_8b.yml --save probes/v_llama3_8b.npy
python -m models.lora_drift --config configs/llama3_8b.yml --save adapters/llama3_8b-lora-drift
