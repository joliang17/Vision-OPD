#!/bin/bash
# 2026-07-21: 本机自己交付的 N4/S2c 以及 301829143 的 QL1，三个都还没人排 eval——
# mlx 之前出过系统性失败(20项批次)，这次直接走本地(conda activate qwen35 让 PATH 上的 vllm
# 本身就是conda版, 不需要 qwen35_shim), 3路并行, 每路1卡
set -uo pipefail
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=$OPSD/Vision-OPD
cd "$V"
LOG=logs/eval_n4_s2c_ql1_driver_301832756.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench
EVAL_VER=server_qwen3vl2b_temp0_4096_generic

source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35

run_lane() { # $1=gpu $2=port $3=model_path $4=model_name
  local gpu=$1 port=$2 mp=$3 mn=$4
  (
    cd "$OPSD/VLMEvalKit"
    env BACKEND=vllm_server PORT=$port EVAL_SETTING_VERSION=$EVAL_VER \
      JUDGE_API_NPROC=2 JUDGE_RETRY=12 REQUEST_TIMEOUT=900 RETRY=4 \
      MODEL_PATH=$mp MODEL_NAME=$mn DATASETS=$DATASETS GPU_IDS=$gpu \
      bash shell_scripts/eval_model_temp0_4096.sh \
      > $V/logs/eval_local_${mn}.log 2>&1
    echo "[$(date)] lane $mn (gpu$gpu) exited rc=$?" >> "$LOG"
  ) &
}

CKPT=$V/checkpoints
log "=== launching N4 + S2c + QL1 local eval (conda qwen35, 3 lanes) ==="
run_lane 0 28920 "$CKPT/Vision-OPD-contrast-standard-uniformweight-Qwen3.5-9B-virl39k-UNFILTERED1img-90step-trial301832756/global_step_90" n4_uniformweight_qwen35_9b_unfiltered_step90
run_lane 1 28921 "$CKPT/Vision-OPD-baseline-seed777-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301832756/global_step_90" s2c_answerhint_seed777_unfiltered_step90
run_lane 2 28922 "$CKPT/Vision-OPD-contrast-standard-uniformweight-Qwen3.5-4B-virl39k-UNFILTERED1img-90step-len4096-trial301829143/global_step_90" ql1_uniform_qwen35_4b_unfiltered_len4096_step90
wait
log "=== all 3 lanes finished ==="
