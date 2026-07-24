#!/bin/bash
# no-anchor 跨 scale: Qwen3-VL-8B. 终局配置 + ra_contrast_anchor_coef=0, 90步, 8卡
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

SUF=${MLX_QUEUE_SUFFIX:-pub}
MODEL_PATH=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b TRAINER_N_GPUS_PER_NODE=8 \
EXPERIMENT_NAME=Vision-OPD-pure-contrast-noanchor-uniform-Qwen3-VL-8B-virl39k-UNFILTERED1img-90step-${SUF} \
ANSWER_VAL_TRAIN_FILE="$V/data/virl39k_train_noimg_unfiltered_1img.parquet" \
TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=1.0 \
  actor_rollout_ref.actor.self_distillation.ra_contrast_anchor_coef=0.0 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90
