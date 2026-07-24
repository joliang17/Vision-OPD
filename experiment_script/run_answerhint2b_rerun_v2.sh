#!/bin/bash
# answerhint-2B 干净重跑v2(tokenizer已按坑#7修复); 等GPU7的base MME-RW泳道结束后接
set -x
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR==8{exit !($1>5000)}'; do sleep 60; done
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
BACKEND=vllm_server PORT=28824 \
MODEL_PATH=$V/checkpoints/Vision-OPD-baseline-Qwen3-VL-2B-Instruct-trial301683547/global_step_62 \
MODEL_NAME=answerhint_2b_step62_clean_rerun \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench GPU_IDS=7 \
bash shell_scripts/eval_model_temp0_4096.sh > $V/logs/eval_answerhint2b_clean_rerun_v2_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "[$(date)] answerhint-2B 干净重跑完成"
