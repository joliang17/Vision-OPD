#!/bin/bash
# Phase 2-核心 / Phase 2a: shuffle control. Uses the real noimg ra_weight distribution but
# randomly permutes which response token gets which weight value (same distribution,
# broken token<->visual-relevance correspondence). Isolates "does the specific visual-relevance
# signal matter" from "does any non-uniform gradient reweighting help".
set -euo pipefail

export EXPERIMENT="${EXPERIMENT:-noimg}"
export MODEL_SIZE="${MODEL_SIZE:-2B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-noimg-shuffled-control-Qwen3-VL-2B-Instruct}"

exec bash "$(dirname "$0")/run_experiment_r_noimg.sh" \
    actor_rollout_ref.actor.self_distillation.ra_weighting_mode=shuffled_control \
    actor_rollout_ref.actor.self_distillation.ra_shuffle_seed="${RA_SHUFFLE_SEED:-12345}" \
    actor_rollout_ref.actor.self_distillation.ra_rollout_reweight=False \
    actor_rollout_ref.actor.self_distillation.ra_divergence_alpha=0.0 \
    "$@"
