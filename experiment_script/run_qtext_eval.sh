#!/bin/bash
# qtext(question vs irrelevant text) step62 评测: 9-benchmark(GPU6) + ZoomBench(GPU6, 端口28810)
# 完成后的case级四象限分析(black版 vs qtext版逐题对比)是这个实验的核心产出
set -x
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
CKPT=$V/checkpoints/Vision-OPD-contrast-standard-qtext-Qwen3-VL-2B-Instruct/global_step_62

cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
BACKEND=vllm_server PORT=28809 \
MODEL_PATH="$CKPT" MODEL_NAME=contrast_standard_qtext_step62 \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
GPU_IDS=6 \
bash shell_scripts/eval_model_temp0_4096.sh \
  > "$V/logs/eval_qtext_step62_$(date +%Y%m%d_%H%M%S).log" 2>&1

cd "$V"
bash scripts/run_zoombench_canonical.sh "$CKPT" contrast_standard_qtext_step62 6 28810 \
  > logs/zoombench_qtext_step62_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "[$(date)] qtext step62 全部评测完成——记得做四象限分析(vs contrast-标准black版)"
