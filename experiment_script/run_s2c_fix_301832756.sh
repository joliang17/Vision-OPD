#!/bin/bash
# S2c retry (2026-07-20): 原始启动崩在 flash_attn ABI 不匹配 —— 本机此前排查 QS1 环境时
# `pip install --user` 装的 torch2.10/transformers5.5 污染了 user-local，被默认 python 优先加载，
# 但系统自带 flash_attn 是编译给旧系统 torch 的，对不上。用 PYTHONNOUSERSITE=1 强制走纯系统栈
# (torch2.8.0+transformers4.57.0+flash_attn2.8.1，已验证互相 ABI 兼容)绕开。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/s2c_driver_301832756.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA_U=$V/data/virl39k_train_noimg_unfiltered_1img.parquet

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

log "=== S2c retry: PYTHONNOUSERSITE=1 (纯系统栈, 绕开 user-local 污染) ==="
wait_gpus_free
S2C_NAME=Vision-OPD-baseline-seed777-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301832756
rm -rf "checkpoints/$S2C_NAME"
S2C_LOG=logs/s2c_seed777_retry_$(date +%Y%m%d_%H%M%S).log
log "launching S2c retry (8卡, PYTHONNOUSERSITE=1): $S2C_NAME"
PYTHONNOUSERSITE=1 MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$S2C_NAME ANSWER_VAL_TRAIN_FILE=$DATA_U TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_baseline.sh \
  data.seed=777 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$S2C_LOG" 2>&1 &
sleep 60
if wait_train "checkpoints/$S2C_NAME" "$S2C_LOG" 90; then
  for st in 30 60 90; do
    dd="checkpoints/$S2C_NAME/global_step_${st}"
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "S2c step${st} merged"
  done
  log "S2c DONE + merged"
else
  log "S2c FAILED again — 看 $S2C_LOG"
fi
log "=== S2c retry driver finished ==="
