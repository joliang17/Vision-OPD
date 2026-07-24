#!/bin/bash
# Launch the R_noimg (RA-VAD noimg) experiment: T_ctrl = text-only (image stripped).
# 4 GPUs by default.
#
#   bash scripts/run_experiment_r_noimg.sh
set -euo pipefail
export EXPERIMENT="${EXPERIMENT:-noimg}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
exec bash "$(dirname "$0")/run_vision_opd_ra_vad.sh" "$@"
