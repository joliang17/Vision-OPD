#!/bin/bash
# Phase 2-核心 第十轮: contrast-sharpened distillation target, STANDARD (non-conservative) config.
#
# Same mechanism as run_experiment_contrast_conservative.sh (see that script + doc 第十轮 for the
# full motivation). Differs from the conservative run only on tilt strength and gating —
# α=1.0 (vs 0.5) and no positive-only position gate — so the pair isolates "how hard can we push
# the tilt". Both runs share ctrl = black image (the noimg ctrl's opening-phrase style confound
# is established and has no upside worth ablating — 第五轮), the plausibility mask (β=0.1), and
# the termination-token exclusion (guards against garbage-token support and silent response
# shortening; removing them tests nothing meaningful).
#
#   bash scripts/run_experiment_contrast_standard.sh
set -euo pipefail

export EXPERIMENT="${EXPERIMENT:-black}"
export MODEL_SIZE="${MODEL_SIZE:-2B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"

exec bash "$(dirname "$0")/run_vision_opd_ra_vad.sh" \
    actor_rollout_ref.actor.self_distillation.ra_target_mode=contrast \
    actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=1.0 \
    actor_rollout_ref.actor.self_distillation.ra_contrast_beta=0.1 \
    actor_rollout_ref.actor.self_distillation.ra_contrast_gate_positive_only=False \
    'actor_rollout_ref.actor.self_distillation.ra_contrast_exclude_token_ids=[151643,151645]' \
    "$@"
