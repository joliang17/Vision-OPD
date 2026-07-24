#!/bin/bash
# 等 保守×sr1-90step retry3 训练进程结束后，merge+评测 step30/60/90
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

NAME="Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-sr1-filtered-90step"

while pgrep -f "trainer.experiment_name=${NAME}$" >/dev/null 2>&1 || pgrep -f "trainer.experiment_name=${NAME} " >/dev/null 2>&1; do
  sleep 30
done
echo "[$(date)] 保守×sr1 retry3 训练进程已结束，开始merge"

for step in 30 60 90; do
  ckpt_dir="checkpoints/${NAME}/global_step_${step}"
  if [ -d "${ckpt_dir}/actor" ] && [ ! -f "${ckpt_dir}/config.json" ]; then
    bash scripts/merge_checkpoint.sh "${ckpt_dir}" > "logs/merge_cons_sr1_retry3_step${step}_$(date +%Y%m%d_%H%M%S).log" 2>&1
  fi
done
echo "[$(date)] merge完成，等空卡评测"

find_free_gpu() {
  for gpu in 0 1 2 3 4 5 6 7; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
    if [ "$used" -lt 5000 ]; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

cd VLMEvalKit
for step in 30 60 90; do
  ckpt_dir="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/${NAME}/global_step_${step}"
  if [ -f "${ckpt_dir}/config.json" ]; then
    GPU=""
    while [ -z "$GPU" ]; do
      GPU=$(find_free_gpu)
      [ -z "$GPU" ] && sleep 20
    done
    BACKEND=vllm_server \
    MODEL_PATH="${ckpt_dir}" \
    MODEL_NAME="task5_cons_sr1_step${step}_server" \
    DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
    GPU_IDS=${GPU} \
    nohup bash shell_scripts/eval_model_temp0_4096.sh \
      > /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/logs/eval_task5_cons_sr1_step${step}_server_$(date +%Y%m%d_%H%M%S).log 2>&1
  fi
done
echo "[$(date)] 保守×sr1 step30/60/90 评测完成"
