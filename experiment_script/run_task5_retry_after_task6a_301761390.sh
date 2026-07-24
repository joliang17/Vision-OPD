#!/usr/bin/env bash
# Retry driver for task5 (GRPO x virl39k, Qwen3.5-4B) on trial 301761390.
#
# Task5's first attempt died instantly: run_experiment_grpo_baseline.sh
# defaults PYTHONNOUSERSITE=1, which hides the user-local transformers 5.5.0
# (the only one that knows model type `qwen3_5`) and falls back to the system
# transformers 4.57.0 -> "KeyError: 'qwen3_5'". The other launch scripts
# (run_vision_opd_ra_vad.sh) default PYTHONNOUSERSITE=0, which is why every
# other Qwen3.5 run worked. Fix: pass PYTHONNOUSERSITE=0 explicitly.
#
# The main driver (run_task456_after_baseline_301761390.sh) skipped ahead to
# task6a when task5 exited after 1 minute (it only waits on the PID, no
# success check). This script waits for the currently-running task6a to
# finish (polling its checkpoint dir + process), then merges task6a's 30/60/90
# checkpoints (doing the main driver's job for it, since the main driver's
# remaining logic already ran past), and finally launches task5 correctly.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG="logs/task5_retry_driver_301761390.log"
QWEN35_MODEL=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B
TASK6A_DIR=checkpoints/Vision-OPD-contrast-standard-Qwen3.5-4B-sr1-filtered-90step-trial301761390

log() { echo "[$(date)] $*" | tee -a "$LOG"; }

log "waiting for task6a to produce global_step_90/actor or for its training procs to disappear"
while true; do
  if [ -d "${TASK6A_DIR}/global_step_90/actor" ]; then
    log "task6a reached step 90"
    break
  fi
  if ! pgrep -f "Vision-OPD-contrast-standard-Qwen3.5-4B-sr1-filtered-90step-trial301761390" > /dev/null 2>&1; then
    log "WARNING: task6a processes gone but no step90 checkpoint -- it may have failed; proceeding to launch task5 anyway (GPUs are free either way)"
    break
  fi
  sleep 120
done

# give the main driver a moment to run its own merge of task6a, then check;
# if it didn't (or failed), do the merges here
sleep 180
for s in 30 60 90; do
  d="${TASK6A_DIR}/global_step_${s}"
  if [ -d "${d}/actor" ] && [ ! -f "${d}/config.json" ]; then
    log "merging task6a step ${s} (main driver didn't)"
    bash scripts/merge_checkpoint.sh "${d}" >> "$LOG" 2>&1
  fi
done

log "launching task5 retry: GRPO x virl39k with PYTHONNOUSERSITE=0"
PYTHONNOUSERSITE=0 \
MODEL_PATH="${QWEN35_MODEL}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-grpo-Qwen3.5-4B-virl39k-filtered-trial301761390 \
  TASK_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/virl39k_train_noimg_filtered_1img.parquet \
  MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_grpo_baseline.sh \
  data.filter_overlong_prompts=True \
  >> logs/qwen35_4b_grpo_virl39k_retry_301761390.log 2>&1 &
TASK5_PID=$!
log "task5 retry pid=${TASK5_PID}"

# fail fast check: if it dies within 10 minutes, log loudly
sleep 600
if kill -0 "${TASK5_PID}" 2>/dev/null; then
  log "task5 retry still alive after 10min (good sign)"
else
  log "ERROR: task5 retry died within 10 minutes -- check logs/qwen35_4b_grpo_virl39k_retry_301761390.log"
  exit 1
fi

while kill -0 "${TASK5_PID}" 2>/dev/null; do sleep 120; done
log "task5 retry exited"
LATEST_STEP=$(cat checkpoints/Vision-OPD-grpo-Qwen3.5-4B-virl39k-filtered-trial301761390/latest_checkpointed_iteration.txt 2>/dev/null || echo "")
if [ -n "${LATEST_STEP}" ] && [ -d "checkpoints/Vision-OPD-grpo-Qwen3.5-4B-virl39k-filtered-trial301761390/global_step_${LATEST_STEP}/actor" ]; then
  bash scripts/merge_checkpoint.sh "checkpoints/Vision-OPD-grpo-Qwen3.5-4B-virl39k-filtered-trial301761390/global_step_${LATEST_STEP}" >> "$LOG" 2>&1
  FORCE=1 bash scripts/prune_checkpoints.sh checkpoints/Vision-OPD-grpo-Qwen3.5-4B-virl39k-filtered-trial301761390 5 >> "$LOG" 2>&1
  log "task5 merged+pruned at step ${LATEST_STEP}"
else
  log "WARNING: task5 has no final checkpoint with actor weights (failed?)"
fi
log "task5 retry driver done"
