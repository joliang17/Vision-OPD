#!/bin/bash
# Launch the OPSD baseline experiment: teacher sees the original image + the
# ground-truth answer hint (answer-hint distillation). 4 GPUs by default.
#
#   bash scripts/run_experiment_baseline.sh
set -euo pipefail
export EXPERIMENT="${EXPERIMENT:-baseline}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
exec bash "$(dirname "$0")/run_vision_opd_ra_vad.sh" "$@"
