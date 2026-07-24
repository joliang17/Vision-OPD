#!/bin/bash
# trial 301832756 (2026-07-20): mlx 队列被并发批次堵塞两次，用户拍板改本地 GPU 跑一批 20 项待评。
# 8/10 单点项已被别的机器跑完；本机接剩下 12 个：N3b/N3c(qwen35 shim+conda) + FC1x5 + FC4x5。
# 沿用已有的 _server_qwen3vl2b_temp0_4096_generic 命名口径（避免与任何残留 mlx 目录撞名）。
set -uo pipefail
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=$OPSD/Vision-OPD
cd "$V"
LOG=logs/local_batch_driver_301832756.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench
EVAL_VER=server_qwen3vl2b_temp0_4096_generic

run_lane() { # $1=gpu $2=port $3=model_path $4=model_name $5=shim(1/0)
  local gpu=$1 port=$2 mp=$3 mn=$4 shim=$5
  (
    cd "$OPSD/VLMEvalKit"
    if [ "$shim" = "1" ]; then
      source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
      conda activate qwen35
      export PATH=$V/scripts/qwen35_shim:$PATH
    fi
    env BACKEND=vllm_server PORT=$port EVAL_SETTING_VERSION=$EVAL_VER \
      JUDGE_API_NPROC=2 JUDGE_RETRY=12 REQUEST_TIMEOUT=900 RETRY=4 \
      MODEL_PATH=$mp MODEL_NAME=$mn DATASETS=$DATASETS GPU_IDS=$gpu \
      bash shell_scripts/eval_model_temp0_4096.sh \
      > $V/logs/local_${mn}.log 2>&1
    echo "[$(date)] lane $mn (gpu$gpu) exited rc=$?" >> "$LOG"
  ) &
}

CKPT=$V/checkpoints
log "=== wave 1: N3b/N3c(shim) + FC1 step30/60/90 + FC4 step30/60 (8 lanes) ==="
run_lane 0 28901 "$CKPT/Vision-OPD-contrast-standard-uniformweight-Qwen3.5-2B-virl39k-UNFILTERED1img-90step-trial301829143/global_step_90" n3b_uniformweight_qwen35_2b_unfiltered_step90 1
run_lane 1 28902 "$CKPT/Vision-OPD-baseline-Qwen3.5-2B-virl39k-UNFILTERED1img-90step-trial301829143/global_step_90" n3c_answerhint_qwen35_2b_unfiltered_step90 1
run_lane 2 28903 "$CKPT/Vision-OPD-contrast-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374/global_step_30" fc1_uniform_unfiltered_step30 0
run_lane 3 28904 "$CKPT/Vision-OPD-contrast-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374/global_step_60" fc1_uniform_unfiltered_step60 0
run_lane 4 28905 "$CKPT/Vision-OPD-contrast-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374/global_step_90" fc1_uniform_unfiltered_step90 0
run_lane 5 28906 "$CKPT/Vision-OPD-baseline-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374/global_step_30" fc4_opsd_unfiltered_step30 0
run_lane 6 28907 "$CKPT/Vision-OPD-baseline-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374/global_step_60" fc4_opsd_unfiltered_step60 0
run_lane 7 28908 "$CKPT/Vision-OPD-baseline-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374/global_step_90" fc4_opsd_unfiltered_step90 0
wait
log "=== wave 1 done, starting wave 2: FC1 step120/150 + FC4 step120/150 (4 lanes) ==="

run_lane 0 28909 "$CKPT/Vision-OPD-contrast-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374/global_step_120" fc1_uniform_unfiltered_step120 0
run_lane 1 28910 "$CKPT/Vision-OPD-contrast-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374/global_step_150" fc1_uniform_unfiltered_step150 0
run_lane 2 28911 "$CKPT/Vision-OPD-baseline-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374/global_step_120" fc4_opsd_unfiltered_step120 0
run_lane 3 28912 "$CKPT/Vision-OPD-baseline-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374/global_step_150" fc4_opsd_unfiltered_step150 0
wait
log "=== all 12 lanes finished ==="
