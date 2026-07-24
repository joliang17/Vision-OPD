#!/bin/bash
# P3 续跑 driver（2026-07-17 05:4x，用户指令：P13 之后跑 120/150 边界扫描）
# P3 由 301829143 起头、step10 暂停让位；checkpoint 是 world_size_4 → 必须 4 卡续跑（沿用原目录名）。
# 剩余 140 步 ~2.7h；save_freq=10 全保留，跑完 merge step90/120/150（E3 评测要 120/150，90 作交叉校验点）。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p3_resume_driver.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
P3=checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-virl39k-150step-trial301829143
P13=checkpoints/Vision-OPD-contrast-beta0-Qwen3-VL-2B-virl39k-90step-trial301783374
step() { cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }

log "waiting for P13 to reach 90"
while [ "$(step "$P13")" -lt 90 ]; do
  ps aux | grep "[m]ain_ppo" | grep -q "contrast-beta0" || { sleep 60; [ "$(step "$P13")" -ge 90 ] && break; log "WARNING: P13 procs gone at step $(step "$P13")，继续等外部处理"; sleep 240; }
  sleep 300
done
while true; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '1,4p' | sort -n | tail -1)
  [ "${m:-999999}" -lt 10000 ] && break
  sleep 60
done
# 防撞：确认 301829143 没有恢复自己续跑（mtime 新鲜度）
age=$(( ( $(date +%s) - $(stat -c %Y "$P3/latest_checkpointed_iteration.txt") ) / 60 ))
if [ "$age" -lt 30 ]; then log "ABORT: P3 checkpoint mtime 仅 ${age}min，疑似别处在跑"; exit 1; fi

log "resuming P3 from step $(step "$P3") on GPU0-3 (original name, world_size_4)"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-standard-Qwen3-VL-2B-virl39k-150step-trial301829143 \
  ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_filtered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=150 \
  > logs/p3_150step_resume_$(date +%Y%m%d_%H%M%S).log 2>&1

if [ "$(step "$P3")" -ge 150 ]; then
  for s in 90 120 150; do
    d="$P3/global_step_${s}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "P3 step${s} merged"
  done
  log "P3 DONE + merged (90/120/150)"
else
  log "ERROR: P3 ended at step $(step "$P3")"
fi
