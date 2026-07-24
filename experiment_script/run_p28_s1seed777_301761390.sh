#!/bin/bash
# driver v4 (2026-07-18 晚, 按 devbox 排班表): attach P28@150 -> FC3免费收尾(不prune,保留60-150全部10档+全merge) -> S1补充seed=777
# 协调: S1(seed1234) 已由 301832790 跑, 本机原 FA5(同seed1234) 改 seed=777 避免三重复, 凑 3-seed mean±std
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p25_p28_driver_301761390.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
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
merge_all() {
  local d=$1
  for dd in "$d"/global_step_*; do
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && \
      bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "merged $(basename "$dd")"
  done
}

log "=== driver v4: attach P28@150 -> FC3免费(全档merge不prune) -> S1-seed777 ==="
P28_NAME=Vision-OPD-contrast-standard-uniformweight-Qwen3.5-4B-virl39k-UNFILTERED1img-150step-trial301761390
P28_LOG=$(ls -t logs/p28v2_retry_*.log | head -1)
if wait_train "checkpoints/$P28_NAME" "$P28_LOG" 150; then
  merge_all "checkpoints/$P28_NAME"
  log "P28v2 DONE; FC3免费达成: 保留 step60-150 全部10档(trainer keep=10 自动滚掉10-50, 正好覆盖曲线重点段), 全部已merge, 未prune"
else
  log "P28v2 FAILED — 看 $P28_LOG"
fi
wait_gpus_free

S1_NAME=Vision-OPD-contrast-uniform-seed777-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301761390
rm -rf "checkpoints/$S1_NAME"
S1_LOG=logs/s1_seed777_$(date +%Y%m%d_%H%M%S).log
log "launching S1-seed777 (8卡, 2B uniform×unfiltered, data.seed=777; seed1234已归301832790)"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$S1_NAME ANSWER_VAL_TRAIN_FILE=$DATA_UNF TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  data.seed=777 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$S1_LOG" 2>&1 &
sleep 60
if wait_train "checkpoints/$S1_NAME" "$S1_LOG" 90; then
  for st in 30 60 90; do
    dd="checkpoints/$S1_NAME/global_step_${st}"
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "S1 step${st} merged"
  done
  log "S1-seed777 DONE + merged"
else
  log "S1-seed777 FAILED (fresh-run only)"
fi
log "=== driver v4 finished ==="
