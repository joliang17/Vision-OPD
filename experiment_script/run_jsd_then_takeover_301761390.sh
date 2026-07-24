#!/usr/bin/env bash
# Replaces run_takeover_301683547_after_task6a.sh (killed at user request to
# re-prioritize). New order after the currently-running T2 GRPO finishes:
#   1. merge+prune T2
#   2. JSD ablation (USER PRIORITY): Qwen3-VL-2B x virl39k, contrast-standard,
#      ra_divergence_alpha=0.5, 90 steps -- runs on the SYSTEM python stack
#      (PYTHONNOUSERSITE=1 = the intact original Qwen3-VL env: torch 2.8 /
#      transformers 4.57 / vllm 0.11, verified 2026-07-16)
#   3. JSD eval: step90 9-benchmark + ZoomBench canonical (also system stack)
#   4. resume the 301683547 takeover sequence: T3a -> T3b -> T6b (Qwen3.5,
#      PYTHONNOUSERSITE=0)
# T1 (conservative-default resume) stays manual -- 301683547 was still alive
# at last check; re-check before ever resuming its checkpoint dir.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG="logs/jsd_then_takeover_driver.log"
QWEN35_MODEL=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B
VIRL39K=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/virl39k_train_noimg_filtered_1img.parquet
SR1=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/data/vision_sr1_47k_noimg_v2_filtered.parquet

log() { echo "[$(date)] $*" | tee -a "$LOG"; }

wait_gpus_free() {
  while true; do
    if ! pgrep -f "verl.trainer.main_ppo" > /dev/null 2>&1 && ! pgrep -f "vllm serve" > /dev/null 2>&1; then
      local maxmem
      maxmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
      [ "${maxmem:-999999}" -lt 10000 ] && return 0
    fi
    sleep 60
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
    log "ERROR: no valid checkpoint after exit (latest='${latest}')"
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

# ---- Stage 0: wait for T2 GRPO (pid 2164104), then merge+prune ----
log "waiting for T2 GRPO (pid 2164104)"
while kill -0 2164104 2>/dev/null; do sleep 120; done
log "T2 exited"
T2_DIR=checkpoints/Vision-OPD-grpo-baseline-default-Qwen3.5-4B-trial301761390
LATEST=$(cat "${T2_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")
if [ -n "${LATEST}" ] && [ -d "${T2_DIR}/global_step_${LATEST}/actor" ]; then
  bash scripts/merge_checkpoint.sh "${T2_DIR}/global_step_${LATEST}" >> "$LOG" 2>&1
  FORCE=1 bash scripts/prune_checkpoints.sh "${T2_DIR}" 5 >> "$LOG" 2>&1
  log "T2 merged+pruned at ${LATEST}"
else
  log "WARNING: T2 left no valid checkpoint"
fi
wait_gpus_free

# ---- Stage 1 (USER PRIORITY): JSD ablation, Qwen3-VL-2B x virl39k ----
# Runs on the CURRENT (Qwen3.5/user) stack: verified 2026-07-16 that
# transformers 5.5 loads Qwen3-VL fine (config/model-class/processor), and
# the system stack is missing tensorboard/termcolor/num2words/ijson (this
# week's fixes all went to user-local), so PYTHONNOUSERSITE=1 would re-hit
# the tensorboard crash. Smoke-test 2 steps first, then the real 90.
JSD_DIR=checkpoints/Vision-OPD-contrast-standard-jsd-Qwen3-VL-2B-virl39k-90step-trial301761390
SMOKE_DIR=checkpoints/Vision-OPD-jsd2b-SMOKETEST-trial301761390
log "Stage 1a: JSD smoke test (2 steps, current Qwen3.5/user stack)"
if run_and_verify "${SMOKE_DIR}" 2 logs/jsd_2b_smoketest.log \
  env MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-jsd2b-SMOKETEST-trial301761390 \
  ANSWER_VAL_TRAIN_FILE="${VIRL39K}" TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_divergence_alpha=0.5 \
  data.filter_overlong_prompts=True trainer.total_training_steps=2; then
  log "smoke test passed"
  rm -rf "${SMOKE_DIR}"
  wait_gpus_free
  JSD_OK=1
else
  log "ERROR: JSD smoke test failed on current stack -- NOT falling back automatically (system stack is missing packages); manual investigation needed. Continuing with takeover queue."
  wait_gpus_free
  JSD_OK=0
fi

if [ "${JSD_OK}" = "1" ] && run_and_verify "${JSD_DIR}" 90 logs/jsd_2b_virl39k_train.log \
  env MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-contrast-standard-jsd-Qwen3-VL-2B-virl39k-90step-trial301761390 \
  ANSWER_VAL_TRAIN_FILE="${VIRL39K}" TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_divergence_alpha=0.5 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90; then
  merge_steps "${JSD_DIR}" 30 60 90
  wait_gpus_free

  # ---- Stage 2: JSD eval (step90 9-bench + ZoomBench), current stack ----
  log "Stage 2: JSD step90 eval"
  (
    cd ../VLMEvalKit
    MODEL_PATH=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-jsd-Qwen3-VL-2B-virl39k-90step-trial301761390/global_step_90 \
    MODEL_NAME=jsd_2b_virl39k_90step_step90 \
    DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
    GPU_IDS=0 \
    bash shell_scripts/eval_via_vllm_server.sh
  ) >> logs/jsd_2b_step90_eval.log 2>&1
  log "JSD 9-bench eval done"
  bash scripts/run_zoombench_canonical.sh \
    "${JSD_DIR}/global_step_90" jsd_2b_virl39k_90step_step90 1 8020 >> logs/jsd_2b_step90_zoombench.log 2>&1
  log "JSD ZoomBench done"
  wait_gpus_free
else
  [ "${JSD_OK}" = "1" ] && log "JSD full training failed -- skipping its eval, continuing with takeover queue"
  wait_gpus_free
fi

# ---- Stage 3: resume 301683547 takeover (T3a -> T3b -> T6b), Qwen3.5 stack ----
T3A_DIR=checkpoints/Vision-OPD-contrast-standard-Qwen3.5-4B-virl39k-filtered-90step-trial301761390
if [ -f "${T3A_DIR}/latest_checkpointed_iteration.txt" ] && [ "$(cat ${T3A_DIR}/latest_checkpointed_iteration.txt)" -ge 90 ]; then
  log "T3a already done, skipping"
else
  run_and_verify "${T3A_DIR}" 90 logs/takeover_t3a_std_virl39k.log \
    env PYTHONNOUSERSITE=0 MODEL_PATH="${QWEN35_MODEL}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
    EXPERIMENT_NAME=Vision-OPD-contrast-standard-Qwen3.5-4B-virl39k-filtered-90step-trial301761390 \
    ANSWER_VAL_TRAIN_FILE="${VIRL39K}" MAX_PROMPT_LENGTH=6144 \
    bash scripts/run_experiment_contrast_standard.sh data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  && merge_steps "${T3A_DIR}" 30 60 90
  wait_gpus_free
fi

T3B_DIR=checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301761390
if [ -f "${T3B_DIR}/latest_checkpointed_iteration.txt" ] && [ "$(cat ${T3B_DIR}/latest_checkpointed_iteration.txt)" -ge 90 ]; then
  log "T3b already done, skipping"
else
  run_and_verify "${T3B_DIR}" 90 logs/takeover_t3b_cons_virl39k.log \
    env PYTHONNOUSERSITE=0 MODEL_PATH="${QWEN35_MODEL}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
    EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301761390 \
    ANSWER_VAL_TRAIN_FILE="${VIRL39K}" MAX_PROMPT_LENGTH=6144 \
    bash scripts/run_experiment_contrast_conservative.sh data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  && merge_steps "${T3B_DIR}" 30 60 90
  wait_gpus_free
fi

T6B_DIR=checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-sr1-filtered-90step-trial301761390
if [ -f "${T6B_DIR}/latest_checkpointed_iteration.txt" ] && [ "$(cat ${T6B_DIR}/latest_checkpointed_iteration.txt)" -ge 90 ]; then
  log "T6b already done, skipping"
else
  run_and_verify "${T6B_DIR}" 90 logs/takeover_t6b_cons_sr1.log \
    env PYTHONNOUSERSITE=0 MODEL_PATH="${QWEN35_MODEL}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
    EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3.5-4B-sr1-filtered-90step-trial301761390 \
    ANSWER_VAL_TRAIN_FILE="${SR1}" MAX_PROMPT_LENGTH=4096 \
    bash scripts/run_experiment_contrast_conservative.sh data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  && merge_steps "${T6B_DIR}" 30 60 90
  wait_gpus_free
fi

log "=== all stages done (T1 conservative-default resume still manual) ==="
