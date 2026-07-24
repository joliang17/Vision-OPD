#!/bin/bash
# FC1 6步 HallusionBench 重推(上次推理大面积 API 失败,预测是 "Failed to obtain answer")
# 用法: bash THIS <gpu> <port_base> <step1> [step2 ...]
# 走 qwen35(eval脚本已改qwen35 python)。每步:删坏预测→serve→infer Hallu→judge。
set -uo pipefail
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=$OPSD/Vision-OPD
FC1=$V/checkpoints/Vision-OPD-contrast-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374
GPU=$1; PORT=$2; shift 2
cd "$OPSD/VLMEvalKit"
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
for s in "$@"; do
  MN=fc1_uniform_unfiltered_step${s}_server
  D=outputs_vllm_curated/${MN}_qwen3vl2b_temp0_4096_generic_eval
  # 删坏的 HallusionBench 预测 + 所有判分缓存,强制重推
  rm -f "$D"/${MN}_HallusionBench*.xlsx "$D"/${MN}_HallusionBench*.pkl \
        "$D"/normal_scoring/*HallusionBench* 2>/dev/null
  echo "[$(date)] === reinfer step$s on GPU$GPU port$PORT ==="
  PORT=$PORT \
  BACKEND=vllm_server \
  MODEL_PATH="$FC1/global_step_$s" \
  MODEL_NAME="$MN" \
  DATASETS=HallusionBench \
  GPU_IDS=$GPU \
  bash shell_scripts/eval_model_temp0_4096.sh
  # 校验:真值落地了吗
  sc=$(ls "$D"/normal_scoring/*HallusionBench*score.csv 2>/dev/null|head -1)
  echo "[$(date)] step$s done. score.csv: ${sc:-无}"
done
echo "=== GPU$GPU 组完成 ==="
