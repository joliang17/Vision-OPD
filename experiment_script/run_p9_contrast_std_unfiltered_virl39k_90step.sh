#!/bin/bash
# P9 (arXiv plan): contrast-standard x UNFILTERED virl39k (single-image 36,039), 90 steps, 2B.
# Purpose: sensitivity check for the 38.3K->14.8K filter used by the main table.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
mkdir -p logs
MODEL_SIZE=2B \
TRAINER_N_GPUS_PER_NODE=8 \
EXPERIMENT_NAME=Vision-OPD-contrast-standard-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-mlxjob \
ANSWER_VAL_TRAIN_FILE="$V/data/virl39k_train_noimg_unfiltered_1img.parquet" \
TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
bash scripts/run_experiment_contrast_standard.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  2>&1 | tee "logs/p9_contrast_std_unfiltered_virl39k_90step_${ARNOLD_TRIAL_ID:-local}.log"
