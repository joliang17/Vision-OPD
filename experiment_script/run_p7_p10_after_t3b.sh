#!/bin/bash
# 301783374（现trial 301832790）: T3b收尾后并行跑 P7 + P10 消融（2026-07-17 认领）
#   P7  = contrast-标准配置只改 ra_contrast_alpha=0.5（gate关） × virl39k 90步, GPU0-3 —— paper α/gating 小节唯一钥匙
#   P10 = contrast-标准 + ra_no_sample_gate=True × virl39k 90步, GPU4-7 —— w_t 分解中间点
# 均为 Qwen3-VL-2B 系统env, batch32/len6144, 与 forward 对照(70.68)同口径
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p7_p10_driver.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA=$V/data/virl39k_train_noimg_filtered_1img.parquet

log "waiting for T3b finish driver to exit"
while pgrep -f "run_t3b""_finish" >/dev/null 2>&1; do sleep 120; done
# 等显存回落
while true; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${m:-999999}" -lt 10000 ] && break
  sleep 60
done
T3B=checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301783374
log "T3b final state: step $(cat $T3B/latest_checkpointed_iteration.txt 2>/dev/null) (merge由t3b_finish driver负责)"

P7_DIR=checkpoints/Vision-OPD-contrast-alpha05-nogate-Qwen3-VL-2B-virl39k-90step-trial301783374
P10_DIR=checkpoints/Vision-OPD-contrast-standard-nosamplegate-Qwen3-VL-2B-virl39k-90step-trial301783374

log "launching P7 (GPU0-3) + P10 (GPU4-7) in parallel"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-alpha05-nogate-Qwen3-VL-2B-virl39k-90step-trial301783374 \
  ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=0.5 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/p7_alpha05_nogate_2b_virl39k_$(date +%Y%m%d_%H%M%S).log 2>&1 &

MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=4,5,6,7 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-standard-nosamplegate-Qwen3-VL-2B-virl39k-90step-trial301783374 \
  ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_no_sample_gate=True \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/p10_nosamplegate_2b_virl39k_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 等两个都到 step90 → merge
for pair in "$P7_DIR:P7" "$P10_DIR:P10"; do
  d=${pair%%:*}; n=${pair##*:}
  while true; do
    s=$(cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    [ "${s:-0}" -ge 90 ] && break
    pgrep -f "$(basename "$d")" >/dev/null 2>&1 || { log "ERROR: $n procs gone at step ${s}"; break; }
    sleep 300
  done
  s=$(cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  if [ "${s:-0}" -ge 90 ]; then
    for st in 30 60 90; do
      dd="$d/global_step_${st}"
      [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "$n step${st} merged"
    done
    log "$n DONE + merged"
  fi
done
log "=== p7_p10 driver finished ==="
