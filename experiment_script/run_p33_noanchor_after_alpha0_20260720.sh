#!/bin/bash
# P33 纯对比 target（no-anchor），链在 α=0 matched baseline 之后（2026-07-20，301832790 认领）：
#   终局配置（uniform×unfiltered×2B×β0.1×EOS豁免×90步×len6144×8卡）+ ra_contrast_anchor_coef=0.0（α仍1.0）
#   → target = softmax(lp_hi − lp_ctrl)，纯对比解码 target（去掉 lp_hi anchor）。save_freq=10 观察是否更早崩。
#   零代码（anchor_coef 是现成 config 参数，actor.py:167 / ra_vad.py:350）。
#   等 α=0 driver 的 ckpt 到 step90 + 8 卡全空后 fresh 启动。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p33_noanchor_driver.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
step() { cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
zombie_sweep() { local g p; for g in 0 1 2 3 4 5 6 7; do for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null); do [ "$p" = "5378" ] && continue; log "zombie kill $p GPU$g"; kill -9 "$p" 2>/dev/null || true; done; done; sleep 5; }
all_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1 | awk '{exit !($1<10000)}'; }

A0=checkpoints/Vision-OPD-contrast-alpha0-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301832790
log "armed: waiting for α=0 baseline to reach step90 (chained sequential run)"
while [ "$(step "$A0")" -lt 90 ]; do sleep 300; done
log "α=0 reached step $(step "$A0"); waiting for 8 GPUs to free"
until all_free; do sleep 120; done
zombie_sweep
until all_free; do sleep 60; done

EXP=Vision-OPD-contrast-noanchor-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301832790
CKPT=checkpoints/$EXP
log "launching P33 no-anchor (8 GPUs, 90 steps, anchor_coef=0)"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$EXP \
  ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_unfiltered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  actor_rollout_ref.actor.self_distillation.ra_contrast_anchor_coef=0.0 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/p33_noanchor_90step_$(date +%Y%m%d_%H%M%S).log 2>&1

if [ "$(step "$CKPT")" -ge 90 ]; then
  for st in 30 60 90; do
    d="$CKPT/global_step_${st}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "P33 step${st} merged"
  done
  log "P33 no-anchor DONE + merged (30/60/90)"
else
  log "ERROR: P33 at step $(step "$CKPT")"
fi
log "=== P33 no-anchor driver finished ==="
