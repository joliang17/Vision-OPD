#!/bin/bash
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V/../VLMEvalKit"
export PATH="$V/scripts/qwen35_shim:$PATH"

rerun() { # gpu port ckpt name
  local gpu=$1 port=$2 ckpt=$3 name=$4
  BACKEND=vllm_server PORT=$port MODEL_PATH="$ckpt" MODEL_NAME="$name" \
  DATASETS=HRBench8K GPU_IDS=$gpu \
  JUDGE_API_NPROC=3 REQUEST_TIMEOUT=900 RETRY=12 \
  bash shell_scripts/eval_model_temp0_4096.sh > "$V/logs/${name}_hr8k_rerun.log" 2>&1
  echo "[$(date)] $name HR8K rerun exited" >> "$V/logs/hr8k_backlog_driver.log"
}

rerun 2 28910 "$V/checkpoints/Vision-OPD-baseline-Qwen3.5-4B-trial301761390/global_step_62" answerhint_qwen35_4b_repo_step62_hr8krerun &
rerun 3 28911 "$V/checkpoints/Vision-OPD-baseline-Qwen3.5-4B-virl39k-filtered-trial301761390/global_step_145" answerhint_qwen35_4b_virl39k_step145_hr8krerun &
rerun 4 28912 "$V/checkpoints/Vision-OPD-visionopd-Qwen3.5-4B-trial301761390/global_step_62" visionopd_qwen35_4b_step62_hr8krerun &
wait
echo "[$(date)] === HR8K backlog batch finished ===" >> "$V/logs/hr8k_backlog_driver.log"
