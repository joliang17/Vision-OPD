#!/bin/bash
# 通用 HallusionBench 重推: bash THIS <gpu> <port> <ckpt_dir_prefix> <mn_prefix> <step1> [step2...]
set -uo pipefail
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
GPU=$1; PORT=$2; CKPTPRE=$3; MNPRE=$4; shift 4
cd "$OPSD/VLMEvalKit"
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
for s in "$@"; do
  MN=${MNPRE}${s}_server
  D=outputs_vllm_curated/${MN}_qwen3vl2b_temp0_4096_generic_eval
  rm -f "$D"/${MN}_HallusionBench*.xlsx "$D"/${MN}_HallusionBench*.pkl "$D"/normal_scoring/*HallusionBench* 2>/dev/null
  echo "[$(date)] === reinfer $MN on GPU$GPU port$PORT ==="
  PORT=$PORT BACKEND=vllm_server MODEL_PATH="${CKPTPRE}${s}" MODEL_NAME="$MN" \
    DATASETS=HallusionBench GPU_IDS=$GPU bash shell_scripts/eval_model_temp0_4096.sh
  sc=$(ls "$D"/normal_scoring/*HallusionBench*score.csv 2>/dev/null|head -1)
  echo "[$(date)] $MN done. score.csv: ${sc:-无}"
done
echo "=== GPU$GPU 组完成 ==="
