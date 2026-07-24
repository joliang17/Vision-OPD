#!/bin/bash
# Phase 2-核心 第十轮: contrast-sharpened distillation target, CONSERVATIVE config.
#
# Motivation: the identical-input EMA teacher differs from the student by ~7e-4 nats/token even
# on the highest-RA-weight tokens (nothing to distill; explains why uniform/logprob/attention
# weighting all score the same downstream), while the teacher's own hi-vs-ctrl gap on those
# tokens is ~1800x larger. This run promotes that gap from a scalar weight to the distillation
# target itself: target = softmax(lp_hi + α (lp_hi − lp_ctrl)) within hi's plausibility set.
# Full derivation + offline pre-check: docs/compare_vaopd_0701.md Phase 2-核心 第十轮.
#
# Conservative choices (each backed by a specific pre-check finding):
#   - ctrl = black image (EXPERIMENT=black): kills the opening-phrase style confound at the root
#     (第五轮), and showed the cleanest/最少 argmax churn in the pre-check.
#   - α=0.5: smallest tilt tested; best localization (49-50% of argmax changes in the top
#     ra_raw quartile).
#   - β=0.1: plausibility set restricted to tokens within 10x of hi's argmax probability.
#   - gate_positive_only=True: tilt only at positive-RA-weight positions.
#   - exclude <|endoftext|>(151643)/<|im_end|>(151645): the pre-check caught the raw tilt
#     boosting early-termination tokens, which would silently shorten responses.
#
#   bash scripts/run_experiment_contrast_conservative.sh
set -euo pipefail

export EXPERIMENT="${EXPERIMENT:-black}"
export MODEL_SIZE="${MODEL_SIZE:-2B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"

exec bash "$(dirname "$0")/run_vision_opd_ra_vad.sh" \
    actor_rollout_ref.actor.self_distillation.ra_target_mode=contrast \
    actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=0.5 \
    actor_rollout_ref.actor.self_distillation.ra_contrast_beta=0.1 \
    actor_rollout_ref.actor.self_distillation.ra_contrast_gate_positive_only=True \
    'actor_rollout_ref.actor.self_distillation.ra_contrast_exclude_token_ids=[151643,151645]' \
    "$@"
