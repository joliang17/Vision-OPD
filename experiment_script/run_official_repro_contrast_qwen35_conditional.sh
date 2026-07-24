#!/bin/bash
# 条件任务: 等 Qwen3.5 官方pipeline复刻(visionopd+vanilla)出结果后判断对齐性——
#   若 visionopd 官方复刻 vstar>=90 且 hrbench-8k>=78.5 (即"差距=prompt风格"实锤)
#   → 用同一官方pipeline测 contrast-标准/保守×Qwen3.5 (标准已ready; 保守等训完+merge)
set -x
MM=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Zooming-without-Zooming/mm-eval
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
KEY_CONF=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/config/key.conf
SUMMARY=$V/logs/repro_results_summary.log
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh

echo "[$(date)] 等 qwen35 官方复刻driver退出..."
while pgrep -f "run_official_repro_qwen35.sh" >/dev/null 2>&1; do sleep 120; done

VSTAR=$(grep "visionopd_qwen35_official_repro vstar" "$SUMMARY" | tail -1 | grep -oE "= [0-9.]+" | tr -d '= ')
HR8K=$(grep "visionopd_qwen35_official_repro hrbench-8k" "$SUMMARY" | tail -1 | grep -oE "= [0-9.]+" | tr -d '= ')
echo "[$(date)] visionopd官方复刻: vstar=$VSTAR hrbench8k=$HR8K"
ALIGNED=$(python3 -c "print(1 if float('${VSTAR:-0}')>=90 and float('${HR8K:-0}')>=78.5 else 0)")
if [ "$ALIGNED" != "1" ]; then
  echo "[$(date)] 未达到对齐阈值(vstar>=90 & hr8k>=78.5), 不自动跑contrast, 需人工判断"
  exit 0
fi
echo "[$(date)] 对齐实锤! 开始用官方pipeline测contrast两个模型"

run_lane() {  # <model_path> <tag> <gpu>
  local MP=$1 TAG=$2 GPU=$3
  while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk -v g=$((GPU+1)) 'NR==g{exit !($1>5000)}'; do sleep 60; done
  cd "$MM"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
  export MKL_SERVICE_FORCE_INTEL=1
  export CUDA_VISIBLE_DEVICES=$GPU
  for BENCH in vstar hrbench-4k hrbench-8k zoom-bench; do
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
" >> "$SUMMARY" 2>&1
  done
}

# contrast-标准: 已ready
run_lane $V/checkpoints/Vision-OPD-contrast-standard-Qwen3.5-4B-trial301683547/global_step_62 contrast_std_qwen35_official_repro 4 &
P1=$!

# contrast-保守: 等训完+merge(301683547在训)
(
  CONS=$V/checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-trial301683547
  echo "[$(date)] 等保守版训练完成+merge..."
  while true; do
    FS=$(cat $CONS/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
    [ "$FS" -ge 62 ] && [ -f "$CONS/global_step_$FS/config.json" ] && break
    sleep 300
  done
  echo "[$(date)] 保守版ready(step $FS)"
  run_lane "$CONS/global_step_$FS" contrast_cons_qwen35_official_repro 5
) &
P2=$!
wait $P1 $P2
echo "[$(date)] contrast×Qwen3.5 官方pipeline评测全部完成"
