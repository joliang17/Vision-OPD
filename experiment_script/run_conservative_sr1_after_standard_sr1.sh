#!/bin/bash
# 等 标准×sr1-retry3(GPU4-7, MAX_PROMPT_LENGTH=4096) 跑完后，补跑 保守×sr1(同样4096)
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

while kill -0 1087015 2>/dev/null; do
  sleep 30
done
echo "[$(date)] 标准×sr1-retry3 结束，开始跑 保守×sr1(4096)"

CUDA_VISIBLE_DEVICES=4,5,6,7 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-sr1-filtered-90step \
  ANSWER_VAL_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/vision_sr1_47k_noimg_v2_filtered.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
  nohup bash scripts/run_experiment_contrast_conservative.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/contrast_conservative_sr1_90step_retry2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID_D=$!
wait $PID_D
echo "[$(date)] 保守×sr1(4096) 跑完"
