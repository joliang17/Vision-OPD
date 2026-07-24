#!/bin/bash
# 任务5第二批（保守版）：等标准版两个进程结束后再跑
# virl39k pid=774663, sr1 retry2 pid=922316
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

while kill -0 774663 2>/dev/null || kill -0 922316 2>/dev/null; do
  sleep 30
done
echo "[$(date)] 标准版两个已结束，开始跑保守版"

CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-virl39k-filtered-90step \
  ANSWER_VAL_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/virl39k_train_noimg_filtered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_conservative.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/contrast_conservative_virl39k_90step_retry_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID_B=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-sr1-filtered-90step \
  ANSWER_VAL_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/vision_sr1_47k_noimg_v2_filtered.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_conservative.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/contrast_conservative_sr1_90step_retry_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID_D=$!

wait $PID_B
wait $PID_D
echo "[$(date)] 保守版两个跑完，任务5全部结束"
