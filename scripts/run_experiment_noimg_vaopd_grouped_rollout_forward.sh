#!/bin/bash
# Stage 1: noimg control with VA-OPD grouped token weighting, rollout reweighting, and forward KL.
set -euo pipefail

export EXPERIMENT="${EXPERIMENT:-noimg}"
export MODEL_SIZE="${MODEL_SIZE:-2B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-noimg-vaopd-grouped-rollout-forward-Qwen3-VL-2B-Instruct}"

exec bash "$(dirname "$0")/run_experiment_r_noimg.sh" \
    actor_rollout_ref.actor.self_distillation.ra_weighting_mode=vaopd_grouped \
    actor_rollout_ref.actor.self_distillation.ra_rollout_reweight=True \
    actor_rollout_ref.actor.self_distillation.ra_rollout_tau=1.0 \
    actor_rollout_ref.actor.self_distillation.ra_divergence_alpha=0.0 \
    "$@"
