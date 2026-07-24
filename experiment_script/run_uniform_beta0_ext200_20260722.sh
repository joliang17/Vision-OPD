#!/bin/bash
# uniform β=0 长训练 90→200（替换 Table2 里 standard 的 β=0，让两臂都是终局 uniform）
# ⚠️ FC1 step90 是 world_size_8 分片 ⇒ 必须 8 卡（用户说的4卡不可行）
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/uniform_beta0_ext200_driver.log
log(){ echo "[$(date)] $*" | tee -a "$LOG"; }
step(){ cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
all_free(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1 | awk '{exit !($1<10000)}'; }
EXP=Vision-OPD-contrast-uniform-beta0-Qwen3-VL-2B-virl39k-UNFILTERED1img-ext200-trial301967423
CKPT=checkpoints/$EXP
[ "$(step $CKPT)" -ne 90 ] && { log "FATAL: 期望起点90，实为 $(step $CKPT)"; exit 1; }
log "armed: 等 8 卡全空(β=0 step100 eval 结束)"
until all_free; do sleep 120; done; sleep 60; until all_free; do sleep 60; done
# 系统 torch（跨环境 resume 需与训练时一致，去掉 conda）
log "启动 uniform β=0 resume 90->200 (8卡)"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$EXP ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_unfiltered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  actor_rollout_ref.actor.self_distillation.ra_contrast_beta=0.0 \
  data.filter_overlong_prompts=True trainer.total_training_steps=200 \
  > logs/uniform_beta0_ext200_$(date +%Y%m%d_%H%M%S).log 2>&1
s=$(step "$CKPT")
if [ "$s" -ge 200 ]; then
  for st in 100 120 150 180 200; do d="$CKPT/global_step_${st}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "step${st} merged"; done
  log "uniform β=0 ext200 DONE"
else log "ERROR: 只到 step $s"; fi
log "=== driver finished ==="
