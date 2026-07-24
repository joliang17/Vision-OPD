#!/bin/bash
# 一有空卡就立刻评测 标准×virl39k 的 step30/60（不等其他实验跑完）
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

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

echo "[$(date)] 等待第一张空卡..."
GPU1=""
while [ -z "$GPU1" ]; do
  GPU1=$(find_free_gpu)
  [ -z "$GPU1" ] && sleep 20
done
echo "[$(date)] 用 GPU${GPU1} 跑 step30"

cd VLMEvalKit
BACKEND=vllm_server \
MODEL_PATH=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_30 \
MODEL_NAME=task5_std_virl39k_step30_server \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
GPU_IDS=${GPU1} \
nohup bash shell_scripts/eval_model_temp0_4096.sh \
  > /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/logs/eval_task5_std_virl39k_step30_server_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID30=$!
cd ..

echo "[$(date)] 等待第二张空卡(跑step60)..."
GPU2=""
while [ -z "$GPU2" ]; do
  GPU2=$(find_free_gpu)
  if [ -n "$GPU2" ] && [ "$GPU2" == "$GPU1" ]; then
    GPU2=""
  fi
  [ -z "$GPU2" ] && sleep 20
done
echo "[$(date)] 用 GPU${GPU2} 跑 step60"

cd VLMEvalKit
BACKEND=vllm_server \
MODEL_PATH=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_60 \
MODEL_NAME=task5_std_virl39k_step60_server \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
GPU_IDS=${GPU2} \
nohup bash shell_scripts/eval_model_temp0_4096.sh \
  > /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/logs/eval_task5_std_virl39k_step60_server_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID60=$!
cd ..

wait $PID30
wait $PID60
echo "[$(date)] 标准×virl39k step30/60 评测完成"
