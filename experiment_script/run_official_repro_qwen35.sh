#!/bin/bash
# Qwen3.5-4B visionopd + vanilla base × 官方pipeline(mm-eval) × vstar/hrbench-4k/hrbench-8k
# 验证 V*(-5.2)/HR8K(-6.8)/MMStar(-8.7) 的差距是否也是prompt风格造成(mmstar缺官方json暂跳过)
# infer 用 conda qwen35 环境(vllm 0.18支持Qwen3.5), judge 用系统python
set -x
MM=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Zooming-without-Zooming/mm-eval
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
KEY_CONF=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/config/key.conf
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh

VOPD35=$V/checkpoints/Vision-OPD-visionopd-Qwen3.5-4B-trial301761390/global_step_62
BASE35=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B

wait_gpu_free() { while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk -v g=$(($1+1)) 'NR==g{exit !($1>5000)}'; do sleep 60; done; }

run_lane() {  # <model_path> <tag> <gpu>
  local MP=$1 TAG=$2 GPU=$3
  wait_gpu_free $GPU
  cd "$MM"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
  export MKL_SERVICE_FORCE_INTEL=1
  export CUDA_VISIBLE_DEVICES=$GPU
  for BENCH in vstar hrbench-4k hrbench-8k; do
    conda run -n qwen35 python3 infer_without_tool.py \
      --benchmark "$BENCH" --model_path "$MP" --model "$TAG" \
      --gpus 1 --temperature 0.7 --seed 42 \
      --gpu_memory_utilization 0.85 --max_model_len 24576 \
      > "$V/logs/repro_infer_${TAG}_${BENCH}_$(date +%Y%m%d_%H%M%S).log" 2>&1
    /usr/bin/python3 judge_azure.py --benchmark "$BENCH" --model "${TAG}_seed42" --key_conf "$KEY_CONF" \
      > "$V/logs/repro_judge_${TAG}_${BENCH}_$(date +%Y%m%d_%H%M%S).log" 2>&1
    /usr/bin/python3 -c "
import json
data = json.load(open('judge/${BENCH}/${TAG}_seed42_answer.json'))
correct = sum(1 for x in data if x.get('judge') == 'Yes')
print(f'${TAG} ${BENCH}: {correct}/{len(data)} = {correct/len(data)*100:.2f}%')
" >> "$V/logs/repro_results_summary.log" 2>&1
  done
}

run_lane "$VOPD35" visionopd_qwen35_official_repro 4 &
P1=$!
run_lane "$BASE35" qwen35_vanilla_official_repro 5 &
P2=$!
wait $P1 $P2
echo "[$(date)] Qwen3.5 官方pipeline复刻完成"
