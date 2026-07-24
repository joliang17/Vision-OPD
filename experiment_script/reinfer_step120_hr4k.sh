#!/bin/bash
set -uo pipefail
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "$OPSD/VLMEvalKit"
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
MN=beta0_matchfc1_step120
D=outputs_vllm_curated/${MN}_qwen3vl2b_temp0_4096_generic_eval
rm -f "$D"/${MN}_HRBench4K*.xlsx "$D"/normal_scoring/*HRBench4K* 2>/dev/null
find "$D" -name "*HRBench4K*" -delete 2>/dev/null
PORT=11271 BACKEND=vllm_server API_NPROC=256 JUDGE_API_NPROC=64 \
  MODEL_PATH="$1" MODEL_NAME="$MN" DATASETS=HRBench4K GPU_IDS=0 \
  bash shell_scripts/eval_model_temp0_4096.sh
echo "=== step120 HR4K 补完 ==="
