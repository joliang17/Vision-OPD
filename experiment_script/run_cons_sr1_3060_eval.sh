#!/bin/bash
set -x
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
NAME=Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-sr1-filtered-90step
DS="BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench"

run_one() {
  local STEP=$1 GPU=$2 PORT=$3
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
  BACKEND=vllm_server MODEL_PATH="$V/checkpoints/$NAME/global_step_${STEP}" \
    MODEL_NAME="cons_sr1_90step_step${STEP}" DATASETS="$DS" GPU_IDS=$GPU \
    bash shell_scripts/eval_model_temp0_4096.sh > "$V/logs/cons_sr1_9bench_step${STEP}_$(date +%Y%m%d_%H%M%S).log" 2>&1
  cd "$V"
  bash scripts/run_zoombench_canonical.sh "checkpoints/$NAME/global_step_${STEP}" "cons_sr1_90step_step${STEP}" $GPU $PORT \
    > "logs/cons_sr1_zoombench_step${STEP}_$(date +%Y%m%d_%H%M%S).log" 2>&1
}

run_one 30 4 8060 &
run_one 60 5 8061 &
run_one 90 6 8062 &
wait
echo "[$(date)] 保守×sr1-90step 30/60/90 全部评测完成"
