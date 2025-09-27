#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/pythia410m.yml}
python -m models.lora_drift --config $CFG --save adapters/pythia410m-lora-drift
