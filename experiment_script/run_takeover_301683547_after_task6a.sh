#!/usr/bin/env bash
# Takeover driver: after this machine's task6a finishes, pick up trial
# 301683547's unfinished Qwen3.5-4B queue (that machine is expected to be
# killed within ~24h of 2026-07-15 18:00).
#
# Safety rules learned from earlier incidents:
# - Never train into a checkpoint dir another live machine is writing
#   (the conservative-virl39k collision, 2026-07-14). Before resuming
#   301683547's conservative-default run, require its checkpoint mtime to be
#   STALE (>60min) — fresh mtime means the machine is still alive and
#   working; skip and log instead.
# - Verify success by checkpoint artifacts, not process exit (both queue
#   drivers independently hit the silent-skip bug on 2026-07-15).
# - New runs started here get the -trial301761390 suffix; only the RESUME of
#   301683547's own partial run keeps its original name (resume_mode=auto
#   picks up its latest step).
set -uo pipefail
cd "$(dirname "$0")/.."
LOG="logs/takeover_301683547_driver.log"
QWEN35_MODEL=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B
TASK6A_DIR=checkpoints/Vision-OPD-contrast-standard-Qwen3.5-4B-sr1-filtered-90step-trial301761390

log() { echo "[$(date)] $*" | tee -a "$LOG"; }

wait_gpus_free() {
  # wait until no ray/vllm training procs AND all GPUs back to <10GB
  while true; do
    if ! pgrep -f "verl.trainer.main_ppo" > /dev/null 2>&1; then
      local maxmem
      maxmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
      if [ "${maxmem:-999999}" -lt 10000 ]; then
        return 0
      fi
    fi
    sleep 60
  done
}

run_and_verify() {
  # run_and_verify <ckpt_dir> <expected_final_step_or_empty> <logfile> <cmd...>
  local ckpt_dir="$1" want_step="$2" logfile="$3"; shift 3
  log "launching: $* (log: ${logfile})"
  nohup "$@" >> "${logfile}" 2>&1 &
  local pid=$!
  sleep 600
  if ! kill -0 "${pid}" 2>/dev/null; then
    log "ERROR: died within 10min, see ${logfile}; NOT retrying automatically"
    return 1
  fi
  log "alive after 10min (pid ${pid}), waiting for exit"
  while kill -0 "${pid}" 2>/dev/null; do sleep 120; done
  local latest
  latest=$(cat "${ckpt_dir}/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")
  if [ -z "${latest}" ] || [ ! -d "${ckpt_dir}/global_step_${latest}/actor" ]; then
    log "ERROR: no valid checkpoint after exit (latest='${latest}'), see ${logfile}"
    return 1
  fi
  if [ -n "${want_step}" ] && [ "${latest}" != "${want_step}" ]; then
    log "WARNING: finished at step ${latest}, expected ${want_step} (crash mid-run?)"
  fi
  bash scripts/merge_checkpoint.sh "${ckpt_dir}/global_step_${latest}" >> "$LOG" 2>&1
  FORCE=1 bash scripts/prune_checkpoints.sh "${ckpt_dir}" 3 >> "$LOG" 2>&1
  log "done + merged: ${ckpt_dir} @ step ${latest}"
  return 0
}

already_done() {
  # a run is "done" if latest_checkpointed_iteration.txt >= expected step and actor exists
  local ckpt_dir="$1" want_step="$2"
  local latest
  latest=$(cat "${ckpt_dir}/latest_checkpointed_iteration.txt" 2>/dev/null || echo "0")
  [ "${latest:-0}" -ge "${want_step}" ] && [ -d "${ckpt_dir}/global_step_${latest}/actor" ]
}

log "=== takeover driver started; waiting for task6a to finish first ==="
while true; do
  if [ -f "${TASK6A_DIR}/latest_checkpointed_iteration.txt" ]; then
    s=$(cat "${TASK6A_DIR}/latest_checkpointed_iteration.txt")
    if [ "${s:-0}" -ge 90 ]; then log "task6a reached step 90"; break; fi
  fi
  if ! pgrep -f "Vision-OPD-contrast-standard-Qwen3.5-4B-sr1-filtered-90step-trial301761390" > /dev/null 2>&1; then
    log "WARNING: task6a procs gone before step90 (failed again?); continuing to takeover anyway"
    break
  fi
  sleep 300
done
# merge task6a's own output if present and unmerged
for s in 30 60 90; do
  d="${TASK6A_DIR}/global_step_${s}"
  if [ -d "${d}/actor" ] && [ ! -f "${d}/config.json" ]; then
    bash scripts/merge_checkpoint.sh "${d}" >> "$LOG" 2>&1
    log "merged task6a step ${s}"
  fi
done
wait_gpus_free
log "GPUs free, starting takeover sequence"

# ---- T1: conservative (default data) -- RESUME 301683547's run if it died mid-way ----
CONS_DIR=checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-trial301683547
if already_done "${CONS_DIR}" 62; then
  log "T1 conservative-default: already completed by 301683547, skipping"
else
  latest_file="${CONS_DIR}/latest_checkpointed_iteration.txt"
  if [ -f "${latest_file}" ]; then
    age_min=$(( ( $(date +%s) - $(stat -c %Y "${latest_file}") ) / 60 ))
  else
    age_min=99999
  fi
  if [ "${age_min}" -lt 60 ]; then
    log "T1 conservative-default: checkpoint mtime only ${age_min}min old -- 301683547 may still be ALIVE and training; SKIPPING to avoid a write collision. Re-run this stage manually once it's confirmed dead."
  else
    log "T1 conservative-default: stale (${age_min}min) or absent -- taking over with resume_mode=auto under 301683547's original EXPERIMENT_NAME"
    run_and_verify "${CONS_DIR}" 62 logs/takeover_cons_default.log \
      env PYTHONNOUSERSITE=0 MODEL_PATH="${QWEN35_MODEL}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3.5-4B-trial301683547 \
      bash scripts/run_experiment_contrast_conservative.sh
    wait_gpus_free
  fi
fi

# ---- T2: GRPO baseline (default data), shortest first among the fresh ones ----
GRPO_DIR=checkpoints/Vision-OPD-grpo-baseline-default-Qwen3.5-4B-trial301761390
if already_done checkpoints/Vision-OPD-grpo-baseline-default-Qwen3.5-4B-trial301683547 65 || already_done "${GRPO_DIR}" 65; then
  log "T2 GRPO-baseline-default: already done somewhere, skipping"
else
  run_and_verify "${GRPO_DIR}" "" logs/takeover_grpo_default.log \
    env PYTHONNOUSERSITE=0 MODEL_PATH="${QWEN35_MODEL}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 EXPERIMENT_NAME=Vision-OPD-grpo-baseline-default-Qwen3.5-4B-trial301761390 \
    bash scripts/run_experiment_grpo_baseline.sh
  wait_gpus_free
fi

# ---- T3a: contrast-standard x virl39k, 90 steps ----
T3A_DIR=checkpoints/Vision-OPD-contrast-standard-Qwen3.5-4B-virl39k-filtered-90step-trial301761390
if already_done checkpoints/Vision-OPD-contrast-standard-Qwen3.5-4B-virl39k-filtered-90step-trial301683547 90 || already_done "${T3A_DIR}" 90; then
  log "T3a: already done somewhere, skipping"
else
  run_and_verify "${T3A_DIR}" 90 logs/takeover_t3a_std_virl39k.log \
    env PYTHONNOUSERSITE=0 MODEL_PATH="${QWEN35_MODEL}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 EXPERIMENT_NAME=Vision-OPD-contrast-standard-Qwen3.5-4B-virl39k-filtered-90step-trial301761390 ANSWER_VAL_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/virl39k_train_noimg_filtered_1img.parquet MAX_PROMPT_LENGTH=6144 \
    bash scripts/run_experiment_contrast_standard.sh data.filter_overlong_prompts=True trainer.total_training_steps=90
  wait_gpus_free
fi

# ---- T3b: contrast-conservative x virl39k, 90 steps ----
T3B_DIR=checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301761390
if already_done checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301683547 90 || already_done "${T3B_DIR}" 90; then
  log "T3b: already done somewhere, skipping"
else
  run_and_verify "${T3B_DIR}" 90 logs/takeover_t3b_cons_virl39k.log \
    env PYTHONNOUSERSITE=0 MODEL_PATH="${QWEN35_MODEL}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301761390 ANSWER_VAL_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/virl39k_train_noimg_filtered_1img.parquet MAX_PROMPT_LENGTH=6144 \
    bash scripts/run_experiment_contrast_conservative.sh data.filter_overlong_prompts=True trainer.total_training_steps=90
  wait_gpus_free
fi

# ---- T6b: contrast-conservative x sr1, 90 steps (use 4096 -- sr1 OOMs at 6144 on Qwen3.5) ----
T6B_DIR=checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-sr1-filtered-90step-trial301761390
if already_done checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-sr1-filtered-90step-trial301683547 90 || already_done "${T6B_DIR}" 90; then
  log "T6b: already done somewhere, skipping"
else
  run_and_verify "${T6B_DIR}" 90 logs/takeover_t6b_cons_sr1.log \
    env PYTHONNOUSERSITE=0 MODEL_PATH="${QWEN35_MODEL}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3.5-4B-sr1-filtered-90step-trial301761390 ANSWER_VAL_TRAIN_FILE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/vision_sr1_47k_noimg_v2_filtered.parquet MAX_PROMPT_LENGTH=4096 \
    bash scripts/run_experiment_contrast_conservative.sh data.filter_overlong_prompts=True trainer.total_training_steps=90
  wait_gpus_free
fi

log "=== takeover driver finished all stages ==="
