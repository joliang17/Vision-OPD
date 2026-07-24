#!/bin/bash
# 在 VLMEvalKit(server模式) 里复刻 Vision-OPD 官方评测配置：
#   temp=0.7 top_p=0.8 top_k=20 presence_penalty=1.5 max_tokens=8192
#   min_pixels=65536 max_pixels=16777216 (官方16.7M像素,基本不缩图——最大嫌疑项)
# 模型: 官方checkpoint(yijiangli step65) + base 4B, benchmark: VStarBench/HRBench4K/HRBench8K
# 对照官方表: VisionOPD-4B V*84.82/HR4K 81.50/HR8K 77.00; base 81.68/78.50/76.25
set -x
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit

VOPD=/mnt/bn/tns-algo-video-vlm-ruby/yijiangli/project/opsd/Vision-OPD/checkpoints/Vision-OPD-Qwen3-VL-4B-Instruct/global_step_65
BASE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17

for pair in "${VOPD}|visionopd4b_officialcfg_vlmevalkit" "${BASE}|qwen3vl4b_base_officialcfg_vlmevalkit"; do
  MP="${pair%%|*}"; TAG="${pair##*|}"
  BACKEND=vllm_server PORT=28801 \
  TEMPERATURE=0.7 TOP_P=0.8 TOP_K=20 PRESENCE_PENALTY=1.5 MAX_NEW_TOKENS=8192 \
  MIN_PIXELS=65536 MAX_PIXELS=16777216 \
  MODEL_PATH="$MP" MODEL_NAME="$TAG" \
  DATASETS=VStarBench,HRBench4K,HRBench8K GPU_IDS=6 \
  bash shell_scripts/eval_model_temp0_4096.sh \
    > "$V/logs/officialcfg_vlmevalkit_${TAG}_$(date +%Y%m%d_%H%M%S).log" 2>&1
done
echo "[$(date)] VLMEvalKit官方配置复刻完成"
