#!/bin/bash
# base-4B / VOPD-4B(官方ckpt) × 官方pipeline × MME-RealWorld(EN)/MME-RealWorld-CN
# 官方表参照: base 63.27/62.92, VOPD 68.02/68.89
set -x
MM=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Zooming-without-Zooming/mm-eval
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
KEY_CONF=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/config/key.conf

VOPD=/mnt/bn/tns-algo-video-vlm-ruby/yijiangli/project/opsd/Vision-OPD/checkpoints/Vision-OPD-Qwen3-VL-4B-Instruct/global_step_65
BASE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17

run_lane() {  # <model_path> <tag> <gpu>
  local MP=$1 TAG=$2 GPU=$3
  cd "$MM"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
  export MKL_SERVICE_FORCE_INTEL=1
  export CUDA_VISIBLE_DEVICES=$GPU
  for BENCH in mme-realworld mme-realworld-cn; do
    python3 infer_without_tool.py \
      --benchmark "$BENCH" --model_path "$MP" --model "$TAG" \
      --gpus 1 --temperature 0.7 --seed 42 \
      --gpu_memory_utilization 0.85 --max_model_len 24576 \
      > "$V/logs/repro_infer_${TAG}_${BENCH}_$(date +%Y%m%d_%H%M%S).log" 2>&1
    python3 judge_azure.py --benchmark "$BENCH" --model "${TAG}_seed42" --key_conf "$KEY_CONF" \
      > "$V/logs/repro_judge_${TAG}_${BENCH}_$(date +%Y%m%d_%H%M%S).log" 2>&1
    python3 -c "
import json
data = json.load(open('judge/${BENCH}/${TAG}_seed42_answer.json'))
correct = sum(1 for x in data if x.get('judge') == 'Yes')
print(f'${TAG} ${BENCH}: {correct}/{len(data)} = {correct/len(data)*100:.2f}%')
" >> "$V/logs/repro_results_summary.log" 2>&1
  done
}

run_lane "$BASE" qwen3vl4b_base_official_repro 7 &
P1=$!
run_lane "$VOPD" visionopd4b_official_repro 0 &
P2=$!
wait $P1 $P2
echo "[$(date)] base/VOPD MME-RW/CN 官方复刻完成"
