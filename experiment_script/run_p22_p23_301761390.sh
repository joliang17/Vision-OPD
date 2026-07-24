#!/bin/bash
# trial 301761390: Stage V 验证门 P22 + P23 并行（2026-07-17 认领）
#   P22 = 2B answer-hint × UNFILTERED virl39k 90步 (P1 配方换数据)      GPU0-3 —— OPSD 行 filter 等价性, 对照 E1 filtered 67.59
#   P23 = 2B contrast-标准 × UNFILTERED virl39k 90步 + data.seed=1234  GPU4-7 —— 换 seed 一致性, 与 P9/X16 三角对照
# P23 是 fresh-run only（data.seed 对 resumed run 无效, Codex 复核）：挂了必须 rm -rf 重跑, 不能 resume。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p22_p23_driver_301761390.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA=$V/data/virl39k_train_noimg_unfiltered_1img.parquet
TRIAL=trial301761390

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

P22_NAME=Vision-OPD-baseline-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-$TRIAL
P23_NAME=Vision-OPD-contrast-standard-seed1234-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-$TRIAL
P22_LOG=logs/p22_answerhint_unfiltered_$(date +%Y%m%d_%H%M%S).log
P23_LOG=logs/p23_seed1234_unfiltered_$(date +%Y%m%d_%H%M%S).log

log "launching P22 (GPU0-3) + P23 (GPU4-7) as $TRIAL"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=$P22_NAME ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_baseline.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$P22_LOG" 2>&1 &
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=4,5,6,7 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=$P23_NAME ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  data.seed=1234 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$P23_LOG" 2>&1 &
sleep 60
wait_train "checkpoints/$P22_NAME" "$P22_LOG" 90 && { merge_steps "checkpoints/$P22_NAME"; log "P22 DONE + merged"; } || log "P22 FAILED — 看 $P22_LOG"
wait_train "checkpoints/$P23_NAME" "$P23_LOG" 90 && { merge_steps "checkpoints/$P23_NAME"; log "P23 DONE + merged (fresh run, seed 生效)"; } || log "P23 FAILED — fresh-run only, 修好后 rm -rf checkpoints/$P23_NAME 重跑, 勿 resume"
log "=== P22/P23 driver finished ==="
