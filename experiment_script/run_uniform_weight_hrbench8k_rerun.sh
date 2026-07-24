#!/bin/bash
# uniform-weight step62 的 HRBench8K 重跑：原评测有 61/800 条 "Failed to obtain answer via API."
# （当时vLLM请求失败，全部计错，把分数从~71.9压到66.38），judge本身没问题，重跑推理即可
# 等 task5 backfill driver 退出后用 GPU5
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD

while pgrep -f "run_task5_missing_evals_backfill.sh" >/dev/null 2>&1; do
  sleep 60
done
echo "[$(date)] backfill 已退出，重跑 uniform-weight HRBench8K (GPU5)"

CKPT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-uniform-weight-Qwen3-VL-2B-Instruct/global_step_62

cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
BACKEND=vllm_server \
MODEL_PATH="${CKPT}" \
MODEL_NAME=uniform_weight_step62_hrbench8k_rerun \
DATASETS=HRBench8K \
GPU_IDS=5 \
bash shell_scripts/eval_model_temp0_4096.sh \
  > /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/logs/eval_uniform_weight_hrbench8k_rerun_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "[$(date)] uniform-weight HRBench8K 重跑完成"
