#!/bin/bash
# 任务5：contrast × 外部数据 90步短训重跑（保留 step60 checkpoint），从0开始，不resume
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

# --- 第一批：标准版优先（GPU0-3跑virl39k，GPU4-7跑sr1） ---
CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step \
  ANSWER_VAL_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/virl39k_train_noimg_filtered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/contrast_standard_virl39k_90step_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID_A=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-sr1-filtered-90step \
  ANSWER_VAL_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/vision_sr1_47k_noimg_v2_filtered.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/contrast_standard_sr1_90step_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID_C=$!

wait $PID_A
wait $PID_C
echo "[$(date)] 第一批(标准×virl39k, 标准×sr1)完成"

# --- 第二批：保守版 ---
CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-virl39k-filtered-90step \
  ANSWER_VAL_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/virl39k_train_noimg_filtered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_conservative.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/contrast_conservative_virl39k_90step_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID_B=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-sr1-filtered-90step \
  ANSWER_VAL_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/vision_sr1_47k_noimg_v2_filtered.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_conservative.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/contrast_conservative_sr1_90step_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID_D=$!

wait $PID_B
wait $PID_D
echo "[$(date)] 第二批(保守×virl39k, 保守×sr1)完成，全部任务5结束"
