#!/bin/bash
# 保守×sr1-90step 之前两次都OOM崩溃（第二次疑似和本机自己的eval脚本抢GPU4撞车），
# 等有4张连续空卡后用 MAX_PROMPT_LENGTH=4096 重跑（从0开始，之前没有任何checkpoint）
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

find_4_free_gpus() {
  free=()
  for gpu in 0 1 2 3 4 5 6 7; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
    if [ "$used" -lt 5000 ]; then
      free+=("$gpu")
    fi
  done
  if [ "${#free[@]}" -ge 4 ]; then
    echo "${free[0]},${free[1]},${free[2]},${free[3]}"
    return 0
  fi
  return 1
}

echo "[$(date)] 等待4张空卡..."
GPUS=""
while [ -z "$GPUS" ]; do
  GPUS=$(find_4_free_gpus)
  [ -z "$GPUS" ] && sleep 30
done
echo "[$(date)] 用 GPU${GPUS} 跑保守×sr1-90step(retry3, 4096)"

CUDA_VISIBLE_DEVICES=${GPUS} TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-sr1-filtered-90step \
  ANSWER_VAL_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/vision_sr1_47k_noimg_v2_filtered.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
  nohup bash scripts/run_experiment_contrast_conservative.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/contrast_conservative_sr1_90step_retry3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID=$!
wait $PID
echo "[$(date)] 保守×sr1-90step(retry3) 结束"
