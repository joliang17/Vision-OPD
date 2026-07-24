#!/bin/bash
# 验证: FC4 step10 用【原始栈】重推 —— run.py/judge 走 /usr/bin/python+shim(transformers4.57), serve仍qwen35
set -uo pipefail
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "$OPSD/VLMEvalKit"
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35   # 仅供 vllm serve 用 PATH
MN=fc4_opsd_unfiltered_step10_verifyorig
D=outputs_vllm_curated/${MN}_qwen3vl2b_temp0_4096_generic_eval
rm -rf "$D"
# 关键:run.py / judge 用 /usr/bin/python + shim
export VLMEVAL_PYEXE=/usr/bin/python
export PYTHONPATH=$OPSD/Vision-OPD/.syspkg_shim
PORT=11151 BACKEND=vllm_server \
  MODEL_PATH="$1" MODEL_NAME="$MN" DATASETS=HallusionBench GPU_IDS=5 \
  bash shell_scripts/eval_model_temp0_4096.sh
sc=$(ls "$D"/normal_scoring/*HallusionBench*score.csv 2>/dev/null|head -1)
echo "验证 score.csv: ${sc:-无}"
