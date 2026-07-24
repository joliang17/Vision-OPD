#!/bin/bash
# 只重推 step10 的 HRBench8K(降并发防8K大图超时),写回 *_step10_refresh 目录
# 用法: bash THIS <ckpt> <model_name> <gpu> <port>
set -uo pipefail
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
CK=$1; MN=$2; GPU=$3; PORT=$4
cd "$OPSD/VLMEvalKit"
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
D=outputs_vllm_curated/${MN}_qwen3vl2b_temp0_4096_generic_eval
# 删掉失败的 HR8K 预测+判分缓存,强制重推
rm -f "$D"/${MN}_HRBench8K*.xlsx "$D"/${MN}_HRBench8K*.pkl "$D"/normal_scoring/*HRBench8K* 2>/dev/null
PORT=$PORT BACKEND=vllm_server API_NPROC=64 JUDGE_API_NPROC=24 \
  MODEL_PATH="$CK" MODEL_NAME="$MN" DATASETS=HRBench8K GPU_IDS=$GPU \
  bash shell_scripts/eval_model_temp0_4096.sh
echo "=== $MN HR8K 重推完成 ==="
