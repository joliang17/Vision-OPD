#!/bin/bash
# P37 α 细曲线（2026-07-21，301832790 认领；本机 8 卡空闲，α=0/P33 已训完只剩 CPU 重判）：
#   终局配置(uniform×unfiltered×2B×β0.1×EOS豁免×90步×len6144×8卡)只改 ra_contrast_alpha。
#   P37a=0.75 / P37b=1.25 / P37c=1.5，串行。与 FA1(0.5)/P26(1.0)/FA2(2.0)+α=0 拼完整 α 曲线。
#   判读：若 0.75/1.0/1.25/1.5 都 66±1 → "α 在[0.75,1.5]不敏感，仅极值掉"；中间也掉 → 尖峰敏感。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p37_alpha_curve_driver.log
log(){ echo "[$(date)] $*" | tee -a "$LOG"; }
step(){ cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
zombie_sweep(){ local g p; for g in 0 1 2 3 4 5 6 7; do for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null); do [ "$p" = "5378" ] && continue; log "zombie kill $p GPU$g"; kill -9 "$p" 2>/dev/null || true; done; done; sleep 5; }
all_free(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1 | awk '{exit !($1<10000)}'; }

run_one(){ # <tag> <alpha>
  local tag=$1 alpha=$2
  local EXP=Vision-OPD-contrast-${tag}-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301832790
  local CKPT=checkpoints/$EXP
  log "waiting 8 GPUs free for $tag (α=$alpha)"
  until all_free; do sleep 120; done
  zombie_sweep; until all_free; do sleep 60; done
  log "launching $tag (α=$alpha, 8 GPUs, 90 steps)"
  MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
    EXPERIMENT_NAME=$EXP \
    ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_unfiltered_1img.parquet \
    TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
    bash scripts/run_experiment_contrast_standard.sh \
    actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
    actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=$alpha \
    data.filter_overlong_prompts=True trainer.total_training_steps=90 \
    > logs/p37_${tag}_$(date +%Y%m%d_%H%M%S).log 2>&1
  if [ "$(step "$CKPT")" -ge 90 ]; then
    for st in 30 60 90; do
      d="$CKPT/global_step_${st}"
      [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "$tag step${st} merged"
    done
    log "$tag DONE + merged"
  else
    log "ERROR: $tag at step $(step "$CKPT")"
  fi
}

run_one alpha075 0.75
run_one alpha125 1.25
run_one alpha15  1.5
log "=== P37 α curve driver finished (all 3) ==="
