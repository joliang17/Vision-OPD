#!/bin/bash
# Phase 2-核心 attention-reweighting variant: replace the logp_hi - logp_ctrl relevance signal
# with attention-to-image-token concentration, computed by a dedicated frozen (non-FSDP) scorer
# model (see verl/trainer/ppo/ra_vad.py attention_to_image_score +
# verl/workers/actor/dp_actor.py _compute_attention_ra_raw / _get_attention_scorer_model).
#
# Motivation (docs/compare_vaopd_0701.md Phase 2-核心 第七/八轮): offline analysis found the
# logprob-gap signal is dominated by opening-phrase/discourse-marker tokens ("user"/"based"/
# "provided"/...), not true visual-content tokens, both before and after RA-VAD training.
# Attention-to-image-token concentration ranks true content tokens (objects/materials/colors)
# far above discourse markers in the same offline check, on two independent rollout sources.
#
# Known cost/risk (read before launching at scale):
#   - Loads a SECOND, frozen, unsharded copy of the base model on every rank just to score
#     attention (flash-attention never materializes attention weights, and extracting them from
#     the live FSDP-sharded teacher hit multiple FSDP1 summon_full_params/param_offload bugs —
#     see the doc section above for the two failed attempts). Extra GPU memory: one full model
#     copy per rank (~4-5GB bf16 for 2B).
#   - Adds a THIRD forward pass per micro-batch (student hi, teacher ctrl, and now this eager-
#     attention scorer forward), run one sample at a time to bound peak memory, so expect a real
#     throughput hit vs the logprob-only pipeline.
#   - output_attentions=True computes+returns ALL decoder layers regardless of
#     ra_attention_num_layers (that only controls how many of the returned layers get averaged
#     into the score, not how many get computed) — real, unavoidable extra compute/memory per
#     sample even for a single micro-batch row.
#   - Verify with a short smoke run (a handful of steps) before trusting a full launch; watch for
#     OOM and check the new ra_vad/ra_raw_mean metric looks sane (nonzero, no NaN/Inf).
#
#   bash scripts/run_experiment_noimg_attention.sh
set -euo pipefail

export EXPERIMENT="${EXPERIMENT:-noimg}"
export MODEL_SIZE="${MODEL_SIZE:-2B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-noimg-attention-Qwen3-VL-2B-Instruct}"

# Qwen3-VL image placeholder token id (config.image_token_id). Override via env if the base
# model changes.
RA_ATTENTION_IMAGE_TOKEN_ID="${RA_ATTENTION_IMAGE_TOKEN_ID:-151655}"
RA_ATTENTION_NUM_LAYERS="${RA_ATTENTION_NUM_LAYERS:-8}"
# Samples per scorer forward call. 1 is the safest default (validated in the 4-GPU smoke run);
# raise only after confirming GPU memory headroom for the prompt/response lengths in use.
RA_ATTENTION_SCORER_BATCH_SIZE="${RA_ATTENTION_SCORER_BATCH_SIZE:-1}"

exec bash "$(dirname "$0")/run_experiment_r_noimg.sh" \
    actor_rollout_ref.actor.self_distillation.ra_weight_source=attention \
    actor_rollout_ref.actor.self_distillation.ra_attention_image_token_id="${RA_ATTENTION_IMAGE_TOKEN_ID}" \
    actor_rollout_ref.actor.self_distillation.ra_attention_num_layers="${RA_ATTENTION_NUM_LAYERS}" \
    actor_rollout_ref.actor.self_distillation.ra_attention_scorer_batch_size="${RA_ATTENTION_SCORER_BATCH_SIZE}" \
    "$@"
