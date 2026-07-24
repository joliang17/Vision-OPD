#!/bin/bash
# API失败污染评测的批量重跑（依据 docs/eval_integrity_registry.md 2026-07-15 扫描）
# 接在 uniform-weight HRBench8K 重跑之后（同GPU5泳道）：
#   contrast-标准-4B step62: HRBench4K+8K (14/16条失败)
#   contrast-保守-4B step62: HRBench4K+8K (15/12条失败)
#   std-sr1-step140: HRBench8K (32条失败)
# 训崩的三个长训checkpoint(保守sr1-443/保守virl39k-437/标准virl39k-437)不重跑——失败本身是模型复读超时的症状
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

while pgrep -f "run_uniform_weight_hrbench8k_rerun.sh" >/dev/null 2>&1; do
  sleep 60
done
echo "[$(date)] uniform-weight重跑已结束，开始4B/step140的API失败重跑 (GPU5)"

run_eval() {  # <ckpt> <tag> <datasets>
  local CKPT=$1 TAG=$2 DS=$3
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
  BACKEND=vllm_server \
  MODEL_PATH="${CKPT}" \
  MODEL_NAME="${TAG}" \
  DATASETS="${DS}" \
  GPU_IDS=5 \
  bash shell_scripts/eval_model_temp0_4096.sh \
    > /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/logs/rerun_${TAG}_$(date +%Y%m%d_%H%M%S).log 2>&1
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
}

run_eval /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-4B-Instruct/global_step_62 \
  std4b_step62_hrbench_rerun HRBench4K,HRBench8K

run_eval /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-contrast-conservative-Qwen3-VL-4B-Instruct/global_step_62 \
  cons4b_step62_hrbench_rerun HRBench4K,HRBench8K

run_eval /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-sr1-filtered/global_step_140 \
  std_sr1_step140_hrbench8k_rerun HRBench8K

echo "[$(date)] API失败重跑全部完成，记得跑 python3 scripts/scan_eval_integrity.py 更新登记 + 回填文档数字"
