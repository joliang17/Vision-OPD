#!/bin/bash
# Launch the R_qvis (RA-VAD qvis) experiment: T_ctrl = original image + generic prompt.
# 4 GPUs by default.
#
#   bash scripts/run_experiment_r_qvis.sh
set -euo pipefail
export EXPERIMENT="${EXPERIMENT:-qvis}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
exec bash "$(dirname "$0")/run_vision_opd_ra_vad.sh" "$@"
