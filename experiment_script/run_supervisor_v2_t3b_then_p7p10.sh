#!/bin/bash
# supervisor v2（2026-07-17 00:5x）：T3b(正在跑最后3步) → merge → 重启 P7/P10（首跑死于系统python
# 丢 tensorboard，已重装）→ merge。存活检测用 ps aux + grep（pgrep -f 对超长 cmdline 会漏匹配）。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/supervisor_v2.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA=$V/data/virl39k_train_noimg_filtered_1img.parquet

P7=checkpoints/Vision-OPD-contrast-alpha05-nogate-Qwen3-VL-2B-virl39k-90step-trial301783374
P10=checkpoints/Vision-OPD-contrast-standard-nosamplegate-Qwen3-VL-2B-virl39k-90step-trial301783374
T3B=checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301783374

step() { cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
alive() { ps aux | grep "[m]ain_ppo" | grep -q "$1"; }
merge90() {
  for s in 30 60 90; do
    d="$1/global_step_${s}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "$2 step${s} merged"
  done
}
wait_gpus() {
  while true; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${m:-999999}" -lt 10000 ] && return 0
    sleep 60
  done
}

# ---- 1. T3b ----
while [ "$(step "$T3B")" -lt 90 ]; do
  alive "conservative-Qwen3.5-4B-virl39k" || { sleep 60; [ "$(step "$T3B")" -ge 90 ] || log "ERROR: T3b 进程消失于 step $(step "$T3B")"; break; }
  sleep 180
done
[ "$(step "$T3B")" -ge 90 ] && { merge90 "$T3B" "T3b"; log "T3b DONE"; }
wait_gpus

# ---- 2. P7 + P10 并行重启 ----
log "launching P7 (GPU0-3) + P10 (GPU4-7)"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-alpha05-nogate-Qwen3-VL-2B-virl39k-90step-trial301783374 \
  ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=0.5 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/p7_alpha05_nogate_retry_$(date +%Y%m%d_%H%M%S).log 2>&1 &

MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=4,5,6,7 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-standard-nosamplegate-Qwen3-VL-2B-virl39k-90step-trial301783374 \
  ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_no_sample_gate=True \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/p10_nosamplegate_retry_$(date +%Y%m%d_%H%M%S).log 2>&1 &

sleep 900   # 给足初始化时间再开始存活检测
for pair in "$P7:P7:alpha05-nogate" "$P10:P10:standard-nosamplegate"; do
  d=$(echo "$pair" | cut -d: -f1); n=$(echo "$pair" | cut -d: -f2); pat=$(echo "$pair" | cut -d: -f3)
  while [ "$(step "$d")" -lt 90 ]; do
    alive "$pat" || { sleep 60; [ "$(step "$d")" -ge 90 ] || { log "ERROR: $n 进程消失于 step $(step "$d")"; break; }; }
    sleep 300
  done
  [ "$(step "$d")" -ge 90 ] && { merge90 "$d" "$n"; log "$n DONE"; }
done
log "=== supervisor v2 finished ==="
