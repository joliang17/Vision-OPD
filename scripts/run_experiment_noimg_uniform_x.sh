#!/bin/bash
# Phase 2-核心 对照组 X: pure EMA self-teacher, no visual-relevance token weighting at all
# (w_t=1 for every response token, sample gate bypassed). Isolates "does EMA self-distillation
# alone already have a regularizing effect, independent of any token weighting."
set -euo pipefail

export EXPERIMENT="${EXPERIMENT:-noimg}"
export MODEL_SIZE="${MODEL_SIZE:-2B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-noimg-uniform-X-Qwen3-VL-2B-Instruct}"

exec bash "$(dirname "$0")/run_experiment_r_noimg.sh" \
    actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
    "$@"
