#!/usr/bin/env bash
# Fix-and-rerun driver for T3a/T3b (auto-fix authorized by user 2026-07-16).
#
# Root cause of both failures: identical 66.31 GiB OOM in update_policy
# backward -- Qwen3.5-4B's 248K vocab x full-vocab distillation doesn't fit
# at MAX_PROMPT_LENGTH=6144 even on virl39k (the 6144 convention came from
# Qwen3-VL whose vocab is 63% smaller; task6a hit the same wall on sr1 and
# was fixed with 4096). Fix: rerun both with MAX_PROMPT_LENGTH=4096.
#
# Also runs the eval-stack discriminator afterwards: re-eval the forward
# control checkpoint's BLINK under the current vllm 0.18 stack to decide
# whether JSD's 16pp drop is real or an eval-stack artifact.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG="logs/t3ab_fix_driver.log"
QWEN35_MODEL=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B
VIRL39K=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/virl39k_train_noimg_filtered_1img.parquet

log() { echo "[$(date)] $*" | tee -a "$LOG"; }

wait_gpus_free() {
  while true; do
    if ! pgrep -f "verl.trainer.main_ppo" > /dev/null 2>&1 && ! pgrep -f "vllm serve" > /dev/null 2>&1; then
      local maxmem
      maxmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
      [ "${maxmem:-999999}" -lt 10000 ] && return 0
    fi
    sleep 120
  done
}

run_and_verify() {
  local ckpt_dir="$1" want_step="$2" logfile="$3"; shift 3
  log "launching: $* (log: ${logfile})"
  nohup "$@" >> "${logfile}" 2>&1 &
  local pid=$!
  sleep 600
  if ! kill -0 "${pid}" 2>/dev/null; then
    log "ERROR: died within 10min, see ${logfile}"
    return 1
  fi
  log "alive after 10min (pid ${pid})"
  while kill -0 "${pid}" 2>/dev/null; do sleep 120; done
  local latest
  latest=$(cat "${ckpt_dir}/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")
  if [ -z "${latest}" ] || [ ! -d "${ckpt_dir}/global_step_${latest}/actor" ]; then
    log "ERROR: no valid checkpoint after exit"
    return 1
  fi
  [ -n "${want_step}" ] && [ "${latest}" != "${want_step}" ] && log "WARNING: stopped at ${latest}, expected ${want_step}"
  return 0
}

merge_steps() {
  local ckpt_dir="$1"; shift
  for s in "$@"; do
    d="${ckpt_dir}/global_step_${s}"
    if [ -d "${d}/actor" ] && [ ! -f "${d}/config.json" ]; then
      bash scripts/merge_checkpoint.sh "${d}" >> "$LOG" 2>&1
      log "merged ${d}"
    fi
  done
}

log "=== T3a/T3b fix driver started; waiting for T6b (and any other training) to finish ==="
wait_gpus_free
log "GPUs free"

# ---- discriminator eval FIRST (cheap, 1 GPU, ~30min): forward control BLINK on vllm 0.18 ----
log "Discriminator: forward-control checkpoint BLINK under current vllm 0.18 stack"
(
  cd ../VLMEvalKit
  MODEL_PATH=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_90 \
  MODEL_NAME=fwdctl_2b_step90_vllm018_discriminator \
  DATASETS=BLINK \
  GPU_IDS=0 \
  bash shell_scripts/eval_via_vllm_server.sh
) >> logs/fwdctl_vllm018_discriminator.log 2>&1
log "discriminator eval done -- compare BLINK vs original 58.18 (vllm 0.11): see logs/fwdctl_vllm018_discriminator.log"

# ---- T3a rerun @ 4096, clean start ----
T3A_DIR=checkpoints/Vision-OPD-contrast-standard-Qwen3.5-4B-virl39k-filtered-90step-trial301761390
rm -rf "${T3A_DIR}"
log "T3a rerun @ MAX_PROMPT_LENGTH=4096 (clean start)"
run_and_verify "${T3A_DIR}" 90 logs/takeover_t3a_std_virl39k_4096.log \
  env PYTHONNOUSERSITE=0 MODEL_PATH="${QWEN35_MODEL}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-contrast-standard-Qwen3.5-4B-virl39k-filtered-90step-trial301761390 \
  ANSWER_VAL_TRAIN_FILE="${VIRL39K}" MAX_PROMPT_LENGTH=4096 \
  bash scripts/run_experiment_contrast_standard.sh data.filter_overlong_prompts=True trainer.total_training_steps=90 \
&& merge_steps "${T3A_DIR}" 30 60 90
wait_gpus_free

# ---- T3b rerun @ 4096, clean start (delete the stale step10 from the 6144 attempt) ----
T3B_DIR=checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301761390
rm -rf "${T3B_DIR}"
log "T3b rerun @ MAX_PROMPT_LENGTH=4096 (clean start, old 6144-trained step10 discarded)"
run_and_verify "${T3B_DIR}" 90 logs/takeover_t3b_cons_virl39k_4096.log \
  env PYTHONNOUSERSITE=0 MODEL_PATH="${QWEN35_MODEL}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301761390 \
  ANSWER_VAL_TRAIN_FILE="${VIRL39K}" MAX_PROMPT_LENGTH=4096 \
  bash scripts/run_experiment_contrast_conservative.sh data.filter_overlong_prompts=True trainer.total_training_steps=90 \
&& merge_steps "${T3B_DIR}" 30 60 90
wait_gpus_free

log "=== T3a/T3b fix driver done ==="
