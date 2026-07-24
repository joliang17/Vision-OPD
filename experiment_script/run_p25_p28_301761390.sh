#!/bin/bash
# trial 301761390: P25 -> P28 chain (both assigned here, both untouched — 2026-07-18 claimed)
#   P25 = Qwen3.5 answer-hint × unfiltered, 90步, len6144   (主表 Qwen3.5 OPSD 行, unfiltered口径)
#   P28 = Qwen3.5 uniform-weight × unfiltered, 90步, len6144 (w_t 三点判定第三点, unfiltered口径, 对齐P16实测len)
# 沿用 T3a/T3b 验证过的配方: PYTHONNOUSERSITE=0 (user-local = Qwen3.5 stack), MODEL_PATH=Qwen3.5-4B snapshot, 8卡
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p25_p28_driver_301761390.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
QWEN35_MODEL=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B
DATA=$V/data/virl39k_train_noimg_unfiltered_1img.parquet

wait_gpus_free() {
  while true; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${m:-999999}" -lt 10000 ] && break
    sleep 60
  done
}
wait_train() {
  local d=$1 f=$2 tgt=$3
  while true; do
    local s; s=$(cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    [ "${s:-0}" -ge "$tgt" ] && return 0
    if [ -f "$f" ]; then
      local age=$(( $(date +%s) - $(stat -c %Y "$f") ))
      [ "$age" -gt 900 ] && ! ps aux | grep -v grep | grep -q verl.trainer.main_ppo && { log "DEAD: $d at step ${s}"; return 1; }
    fi
    sleep 300
  done
}
merge_steps() {
  for st in 30 60 90; do
    local dd="$1/global_step_${st}"
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && \
      bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "$(basename "$1") step${st} merged"
  done
}

log "=== P25/P28 chain start ==="
wait_gpus_free

# ---- P25: Qwen3.5 answer-hint x unfiltered ----
P25_NAME=Vision-OPD-baseline-Qwen3.5-4B-virl39k-UNFILTERED1img-90step-trial301761390
P25_LOG=logs/p25_answerhint_qwen35_unfiltered_$(date +%Y%m%d_%H%M%S).log
log "launching P25: $P25_NAME"
env PYTHONNOUSERSITE=0 MODEL_PATH="$QWEN35_MODEL" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$P25_NAME ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_baseline.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$P25_LOG" 2>&1 &
sleep 60
if wait_train "checkpoints/$P25_NAME" "$P25_LOG" 90; then
  merge_steps "checkpoints/$P25_NAME"; log "P25 DONE + merged"
else
  log "P25 FAILED — 看 $P25_LOG, 继续尝试 P28"
fi
wait_gpus_free

# ---- P28: Qwen3.5 uniform-weight x unfiltered ----
P28_NAME=Vision-OPD-contrast-standard-uniformweight-Qwen3.5-4B-virl39k-UNFILTERED1img-90step-trial301761390
P28_LOG=logs/p28_uniformweight_qwen35_unfiltered_$(date +%Y%m%d_%H%M%S).log
log "launching P28: $P28_NAME"
env PYTHONNOUSERSITE=0 MODEL_PATH="$QWEN35_MODEL" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$P28_NAME ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$P28_LOG" 2>&1 &
sleep 60
if wait_train "checkpoints/$P28_NAME" "$P28_LOG" 90; then
  merge_steps "checkpoints/$P28_NAME"; log "P28 DONE + merged"
else
  log "P28 FAILED — 看 $P28_LOG"
fi
log "=== P25/P28 chain finished ==="
