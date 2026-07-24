#!/bin/bash
# P28v2 + FA5 重启 (2026-07-18 16:0x): 首次失败=启动时GPU0被残留进程占用(free 13.8GB<需124.85GB), 现已全空
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p25_p28_driver_301761390.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
QWEN35_MODEL=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B
DATA_UNF=$V/data/virl39k_train_noimg_unfiltered_1img.parquet
wait_gpus_free() { while true; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "${m:-999999}" -lt 5000 ] && break; sleep 60; done; }
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
merge_steps() { local d=$1; shift; for st in "$@"; do local dd="$d/global_step_${st}"; [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "$(basename "$d") step${st} merged"; done; }

log "=== retry driver: P28v2@150 -> FA5 ==="
wait_gpus_free
P28_NAME=Vision-OPD-contrast-standard-uniformweight-Qwen3.5-4B-virl39k-UNFILTERED1img-150step-trial301761390
rm -rf "checkpoints/$P28_NAME"   # 上次 step0 残留
P28_LOG=logs/p28v2_retry_$(date +%Y%m%d_%H%M%S).log
log "launching P28v2 retry: $P28_NAME"
env PYTHONNOUSERSITE=0 MODEL_PATH="$QWEN35_MODEL" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$P28_NAME ANSWER_VAL_TRAIN_FILE=$DATA_UNF TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  data.filter_overlong_prompts=True trainer.total_training_steps=150 \
  > "$P28_LOG" 2>&1 &
sleep 60
if wait_train "checkpoints/$P28_NAME" "$P28_LOG" 150; then
  merge_steps "checkpoints/$P28_NAME" 30 60 90 120 150; log "P28v2 DONE + merged"
else
  log "P28v2 FAILED again — 看 $P28_LOG"
fi
wait_gpus_free
FA5_NAME=Vision-OPD-contrast-uniformweight-seed1234-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301761390
rm -rf "checkpoints/$FA5_NAME"
FA5_LOG=logs/fa5_retry_$(date +%Y%m%d_%H%M%S).log
log "launching FA5 retry (8卡)"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$FA5_NAME ANSWER_VAL_TRAIN_FILE=$DATA_UNF TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  data.seed=1234 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$FA5_LOG" 2>&1 &
sleep 60
wait_train "checkpoints/$FA5_NAME" "$FA5_LOG" 90 && { merge_steps "checkpoints/$FA5_NAME" 30 60 90; log "FA5 DONE + merged"; } || log "FA5 FAILED again"
log "=== retry driver finished ==="
