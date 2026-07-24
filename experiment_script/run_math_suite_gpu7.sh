#!/bin/bash
# GPU7: virl39k-90step contrast-标准(step90) 和 Qwen3-VL-2B-Instruct base
# 在 WeMath/MathVista_MINI/MathVerse_MINI/MMMU_DEV_VAL/OCRBench 五个benchmark上的评测(串行)
set -x
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
DS="WeMath,MathVista_MINI,MathVerse_MINI,MMMU_DEV_VAL,OCRBench"

cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit

BACKEND=vllm_server \
MODEL_PATH=$V/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_90 \
MODEL_NAME=std_virl39k_90step_step90_mathsuite \
DATASETS=$DS GPU_IDS=7 \
bash shell_scripts/eval_model_temp0_4096.sh \
  > $V/logs/mathsuite_std_virl39k_step90_$(date +%Y%m%d_%H%M%S).log 2>&1

BACKEND=vllm_server \
MODEL_PATH=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/transformers/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203 \
MODEL_NAME=qwen3vl2b_base_mathsuite \
DATASETS=$DS GPU_IDS=7 \
bash shell_scripts/eval_model_temp0_4096.sh \
  > $V/logs/mathsuite_qwen3vl2b_base_$(date +%Y%m%d_%H%M%S).log 2>&1

echo "[$(date)] math suite 两个模型评测完成"
