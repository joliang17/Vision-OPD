#!/bin/bash
# 等 task5 相关训练/评测全部结束、8卡都空出来后，跑 grpo-virl39k-filtered 3epoch(8卡)
# save_freq=100, max_actor_ckpt_to_keep=3（只保留最近3个），用户已确认这个取舍
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

all_gpus_free() {
  for gpu in 0 1 2 3 4 5 6 7; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
    if [ "$used" -ge 5000 ]; then
      return 1
    fi
  done
  return 0
}

echo "[$(date)] 等待全部8张卡空闲..."
while ! all_gpus_free; do
  sleep 30
done
echo "[$(date)] 8卡已全部空闲，启动 grpo-virl39k-filtered 3epoch"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-grpo-Qwen3-VL-2B-Instruct-virl39k-filtered-3ep \
  TASK_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/virl39k_train_noimg_filtered.parquet \
  TRAINER_TOTAL_EPOCHS=3 \
  TRAINER_SAVE_FREQ=100 \
  TRAINER_MAX_ACTOR_CKPT_TO_KEEP=3 \
  nohup bash scripts/run_experiment_grpo_baseline.sh \
  > logs/grpo_virl39k_filtered_2b_3ep_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID=$!
wait $PID
echo "[$(date)] grpo-virl39k-filtered 3epoch 训练结束"
