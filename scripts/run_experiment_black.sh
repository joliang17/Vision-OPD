#!/bin/bash
# Launch the black control experiment: T_ctrl = same-size solid-black image.
# 4 GPUs by default.
#
#   bash scripts/run_experiment_black.sh
set -euo pipefail
export EXPERIMENT="${EXPERIMENT:-black}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
exec bash "$(dirname "$0")/run_vision_opd_ra_vad.sh" "$@"
