#!/bin/bash
# 通用 β=0 点 7-bench eval(原始栈): bash THIS <ckpt_dir> <model_name> <gpu> <port> [wait_merge_dir]
set -uo pipefail
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
CKPT=$1; MN=$2; GPU=$3; PORT=$4; WAITM=${5:-}
# 若指定 wait_merge_dir,等它 merge 完(出现 config.json)
if [ -n "$WAITM" ]; then
  for i in $(seq 1 120); do [ -f "$WAITM/config.json" ] && break; sleep 15; done
  [ -f "$WAITM/config.json" ] || { echo "FATAL: $WAITM 未 merge 完"; exit 1; }
fi
cd "$OPSD/VLMEvalKit"
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35   # 仅 serve;run.py/judge 走脚本默认 /usr/bin/python+shim(原始栈)
PORT=$PORT BACKEND=vllm_server MODEL_PATH="$CKPT" MODEL_NAME="$MN" \
  DATASETS=BLINK,MMStar,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,HallusionBench \
  GPU_IDS=$GPU bash shell_scripts/eval_model_temp0_4096.sh
echo "=== $MN eval DONE ==="
