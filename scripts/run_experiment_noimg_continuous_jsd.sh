#!/bin/bash
# Stage 1: noimg control with current continuous RA weighting and generalized JSD.
set -euo pipefail

export EXPERIMENT="${EXPERIMENT:-noimg}"
export MODEL_SIZE="${MODEL_SIZE:-2B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-noimg-continuous-jsd-Qwen3-VL-2B-Instruct}"

exec bash "$(dirname "$0")/run_experiment_r_noimg.sh" \
    actor_rollout_ref.actor.self_distillation.ra_weighting_mode=continuous \
    actor_rollout_ref.actor.self_distillation.ra_rollout_reweight=False \
    actor_rollout_ref.actor.self_distillation.ra_divergence_alpha=0.5 \
    "$@"
