#!/bin/bash
# vanilla Qwen3.5-4B(未训练) VLMEvalKit 9-bench + ZoomBench(conda shim), 等geo3k训练结束后用GPU6
set -x
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
SHIM=$V/scripts/qwen35_shim
BASE35=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B
DS="BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench"

while pgrep -f "run_geo3k_contrast_4b.sh" >/dev/null 2>&1; do sleep 120; done
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
PATH=$SHIM:$PATH BACKEND=vllm_server PORT=28851 \
MODEL_PATH=$BASE35 MODEL_NAME=vanilla_qwen35_4b DATASETS=$DS GPU_IDS=6 \
bash shell_scripts/eval_model_temp0_4096.sh > $V/logs/eval_vanilla_qwen35_$(date +%Y%m%d_%H%M%S).log 2>&1
cd $V
PATH=$SHIM:$PATH bash scripts/run_zoombench_canonical.sh "$BASE35" vanilla_qwen35_4b 6 28852 \
  > logs/zoombench_vanilla_qwen35_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "[$(date)] vanilla Qwen3.5-4B VLMEvalKit评测完成"
