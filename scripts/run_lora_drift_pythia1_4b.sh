#!/usr/bin/env bash
set -euo pipefail
python -m models.probe --config configs/pythia1_4b.yml --save probes/v_pythia1_4b.npy
python -m models.lora_drift --config configs/pythia1_4b.yml --save adapters/pythia1_4b-lora-drift
