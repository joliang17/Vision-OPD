#!/bin/bash
# Launch the R_res (RA-VAD degrade) experiment: T_ctrl = degraded original image.
# Requires data/train_degraded.parquet (run scripts/prepare_degraded_images.py first).
# 4 GPUs by default.
#
#   python3 scripts/prepare_degraded_images.py --input data/train.parquet --output data/train_degraded.parquet
#   bash scripts/run_experiment_r_res.sh
set -euo pipefail
export EXPERIMENT="${EXPERIMENT:-degrade}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
exec bash "$(dirname "$0")/run_vision_opd_ra_vad.sh" "$@"
