#!/bin/bash
# β=0 step120 eval 填崩溃曲线 —— 只跑本周7-bench口径
set -uo pipefail
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
CKPT=$OPSD/Vision-OPD/checkpoints/Vision-OPD-contrast-beta0-Qwen3-VL-2B-virl39k-ext100-trial301967423/global_step_120
cd "$OPSD/VLMEvalKit"
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35   # 仅 vllm serve 用 PATH;run.py/judge 由脚本默认走 /usr/bin/python+shim(原始栈,tf4.57)
BACKEND=vllm_server \
MODEL_PATH="$CKPT" \
MODEL_NAME=beta0_2b_step120 \
DATASETS=BLINK,MMStar,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,HallusionBench \
GPU_IDS=0 \
bash shell_scripts/eval_model_temp0_4096.sh
