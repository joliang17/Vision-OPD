#!/bin/bash
# Launch the Vision-OPD (published) experiment: teacher sees the bbox crop.
# Default: 4 GPUs. Override with TRAINER_N_GPUS_PER_NODE / ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE.
#
#   bash scripts/run_experiment_visionopd.sh            # 4 GPUs, default settings
#   bash scripts/run_experiment_visionopd.sh trainer.total_epochs=2
set -euo pipefail
export EXPERIMENT="${EXPERIMENT:-visionopd}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
exec bash "$(dirname "$0")/run_vision_opd_ra_vad.sh" "$@"
