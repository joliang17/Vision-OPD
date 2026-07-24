#!/usr/bin/env bash
# Driver for trial_id=301761390's assigned queue (task4 -> task5 -> task6a),
# chained after the pre-existing repo-data baseline finishes.
# Polls (not `wait`, since this script's shell can't wait on another shell's
# child PID) for each stage's final checkpoint, merges+prunes it, then
# launches the next stage.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG="logs/task456_301761390_driver.log"
QWEN35_MODEL=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B

log() { echo "[$(date)] $*" | tee -a "$LOG"; }

wait_for_pid() {
  local pid="$1" label="$2"
  log "waiting for ${label} (pid ${pid}) to exit"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  log "${label} (pid ${pid}) exited"
}

merge_prune() {
  local run_dir="$1" step="$2" keep="$3"
  if [ -d "${run_dir}/global_step_${step}/actor" ]; then
    bash scripts/merge_checkpoint.sh "${run_dir}/global_step_${step}" >> "$LOG" 2>&1
    FORCE=1 bash scripts/prune_checkpoints.sh "${run_dir}" "${keep}" >> "$LOG" 2>&1
    log "merged+pruned ${run_dir} at step ${step}"
  else
    log "WARNING: ${run_dir}/global_step_${step}/actor missing, skipping merge"
  fi
}

# ---- Stage 0: wait for the pre-existing repo-data baseline ----
wait_for_pid 874854 "pre-existing repo-data baseline"
merge_prune checkpoints/Vision-OPD-baseline-Qwen3.5-4B-trial301761390 62 3

# ---- Stage 0.5: rerun HRBench8K for visionopd-Qwen3.5-4B (301783374's integrity scan
# found 56/800 failed samples in the original run, estimated true score ~78.5 vs the
# reported 73.00 -- needs a Qwen3.5-env machine, 1 GPU, ~30min) ----
log "launching HRBench8K rerun for visionopd-Qwen3.5-4B (data-quality fix, GPU0 only)"
(
  cd ../VLMEvalKit
  MODEL_PATH=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-visionopd-Qwen3.5-4B-trial301761390/global_step_62 \
  MODEL_NAME=visionopd_qwen35_4b_trial301761390_hrbench8k_rerun \
  DATASETS=HRBench8K \
  GPU_IDS=0 \
  bash shell_scripts/eval_via_vllm_server.sh
) >> logs/visionopd_qwen35_4b_hrbench8k_rerun_301761390.log 2>&1
log "HRBench8K rerun for visionopd-Qwen3.5-4B done, see logs/visionopd_qwen35_4b_hrbench8k_rerun_301761390.log"

# ---- Stage 1 (task4): baseline (answer-hint) x virl39k, no step limit ----
log "launching task4: baseline x virl39k"
MODEL_PATH="${QWEN35_MODEL}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-baseline-Qwen3.5-4B-virl39k-filtered-trial301761390 \
  ANSWER_VAL_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/virl39k_train_noimg_filtered_1img.parquet \
  MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_baseline.sh \
  data.filter_overlong_prompts=True \
  >> logs/qwen35_4b_baseline_virl39k_301761390.log 2>&1 &
TASK4_PID=$!
log "task4 pid=${TASK4_PID}"
wait_for_pid "${TASK4_PID}" "task4 (baseline x virl39k)"
LATEST_STEP=$(cat checkpoints/Vision-OPD-baseline-Qwen3.5-4B-virl39k-filtered-trial301761390/latest_checkpointed_iteration.txt 2>/dev/null || echo "")
if [ -n "${LATEST_STEP}" ]; then
  merge_prune checkpoints/Vision-OPD-baseline-Qwen3.5-4B-virl39k-filtered-trial301761390 "${LATEST_STEP}" 5
fi

# ---- Stage 2 (task5): GRPO x virl39k, no step limit ----
log "launching task5: GRPO x virl39k"
MODEL_PATH="${QWEN35_MODEL}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-grpo-Qwen3.5-4B-virl39k-filtered-trial301761390 \
  TASK_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/virl39k_train_noimg_filtered_1img.parquet \
  MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_grpo_baseline.sh \
  data.filter_overlong_prompts=True \
  >> logs/qwen35_4b_grpo_virl39k_301761390.log 2>&1 &
TASK5_PID=$!
log "task5 pid=${TASK5_PID}"
wait_for_pid "${TASK5_PID}" "task5 (GRPO x virl39k)"
LATEST_STEP=$(cat checkpoints/Vision-OPD-grpo-Qwen3.5-4B-virl39k-filtered-trial301761390/latest_checkpointed_iteration.txt 2>/dev/null || echo "")
if [ -n "${LATEST_STEP}" ]; then
  merge_prune checkpoints/Vision-OPD-grpo-Qwen3.5-4B-virl39k-filtered-trial301761390 "${LATEST_STEP}" 5
fi

# ---- Stage 3 (task6a): contrast-standard x sr1, 90 steps ----
log "launching task6a: contrast-standard x sr1, 90 steps"
MODEL_PATH="${QWEN35_MODEL}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-contrast-standard-Qwen3.5-4B-sr1-filtered-90step-trial301761390 \
  ANSWER_VAL_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/vision_sr1_47k_noimg_v2_filtered.parquet \
  MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  >> logs/qwen35_4b_contrast_standard_sr1_90step_301761390.log 2>&1 &
TASK6A_PID=$!
log "task6a pid=${TASK6A_PID}"
wait_for_pid "${TASK6A_PID}" "task6a (contrast-standard x sr1, 90step)"
for s in 30 60 90; do
  merge_prune checkpoints/Vision-OPD-contrast-standard-Qwen3.5-4B-sr1-filtered-90step-trial301761390 "${s}" 3
done

log "all of task4/5/6a done"
