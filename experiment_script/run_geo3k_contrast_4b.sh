#!/bin/bash
# Geometry3K × contrast-标准/保守 × Qwen3-VL-4B 正式训练(与VA-OPD对比用)
# 冒烟已通过(batch24×3卡, step1-2指标健康)。正式配置: batch32×4卡(GPU1,2,3,6),与virl39k系列可比
# 等 qtext 评测(GPU6)结束后自动启动; 标准→保守 串行; 1 epoch≈66步,save_freq=10全保留
set -x
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"

echo "[$(date)] 等待 qtext 评测(GPU6) 和 Qwen3.5欠账评测(GPU1-3) 全部结束(评测优先,用户指示)..."
while pgrep -f "run_qtext_eval.sh" >/dev/null 2>&1 || pgrep -f "run_qwen35_evals_conda.sh" >/dev/null 2>&1; do sleep 60; done
echo "[$(date)] GPU1,2,3,6已空,启动 geometry3k contrast-标准-4B"

CUDA_VISIBLE_DEVICES=1,2,3,6 TRAINER_N_GPUS_PER_NODE=4 \
  MODEL_SIZE=4B \
  EXPERIMENT_NAME=Vision-OPD-contrast-standard-Qwen3-VL-4B-Instruct-geometry3k \
  ANSWER_VAL_TRAIN_FILE=$V/data/geometry3k_train.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
  bash scripts/run_experiment_contrast_standard.sh \
  data.filter_overlong_prompts=True \
  > logs/geo3k_std4b_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "[$(date)] 标准版结束,启动保守版"

CUDA_VISIBLE_DEVICES=1,2,3,6 TRAINER_N_GPUS_PER_NODE=4 \
  MODEL_SIZE=4B \
  EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3-VL-4B-Instruct-geometry3k \
  ANSWER_VAL_TRAIN_FILE=$V/data/geometry3k_train.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
  bash scripts/run_experiment_contrast_conservative.sh \
  data.filter_overlong_prompts=True \
  > logs/geo3k_cons4b_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "[$(date)] geometry3k 两个4B训练全部完成(merge/eval另行安排)"
