#!/bin/bash
# retry: WeMath/MathVerse/MMMU/OCRBench 已手动下载进共享LMUData缓存
# std_virl39k step90 只差这4个(MathVista已有); base模型第一次因端口撞车整批失败,全量重跑
set -x
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
DS_MISSING="WeMath,MathVerse_MINI,MMMU_DEV_VAL,OCRBench"
DS_ALL="WeMath,MathVista_MINI,MathVerse_MINI,MMMU_DEV_VAL,OCRBench"

cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit

BACKEND=vllm_server PORT=18761 \
MODEL_PATH=$V/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_90 \
MODEL_NAME=std_virl39k_90step_step90_mathsuite \
DATASETS=$DS_MISSING GPU_IDS=7 \
bash shell_scripts/eval_model_temp0_4096.sh \
  > $V/logs/mathsuite_std_virl39k_step90_retry_$(date +%Y%m%d_%H%M%S).log 2>&1

BACKEND=vllm_server PORT=18762 \
MODEL_PATH=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/transformers/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203 \
MODEL_NAME=qwen3vl2b_base_mathsuite \
DATASETS=$DS_ALL GPU_IDS=7 \
bash shell_scripts/eval_model_temp0_4096.sh \
  > $V/logs/mathsuite_qwen3vl2b_base_retry_$(date +%Y%m%d_%H%M%S).log 2>&1

echo "[$(date)] math suite retry 完成"
