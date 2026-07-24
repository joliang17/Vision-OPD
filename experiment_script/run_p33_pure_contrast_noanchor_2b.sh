#!/bin/bash
# P33: pure-contrast (no-anchor) target ablation. target=softmax(lp_hi - lp_ctrl) via anchor_coef=0.
# Everything else identical to terminal ours-uniform 2B (uniform weight, β=0.1, EOS-exempt, unfiltered, 90步).
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"; mkdir -p logs
MODEL_SIZE=2B TRAINER_N_GPUS_PER_NODE=8 \
EXPERIMENT_NAME=Vision-OPD-pure-contrast-noanchor-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial${ARNOLD_TRIAL_ID:-local} \
ANSWER_VAL_TRAIN_FILE="$V/data/virl39k_train_noimg_unfiltered_1img.parquet" \
TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=1.0 \
  actor_rollout_ref.actor.self_distillation.ra_contrast_anchor_coef=0.0 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  2>&1 | tee "logs/p33_pure_contrast_noanchor_$(date +%Y%m%d_%H%M%S).log"
