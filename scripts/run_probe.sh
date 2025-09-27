#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/pythia410m.yml}
python -m models.probe --config $CFG --save probes/v_layer.pt
