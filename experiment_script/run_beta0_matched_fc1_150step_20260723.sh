#!/bin/bash
# β=0 崩溃臂,完全对齐 FC1(contrast-uniform UNFILTERED 150step)的设置,只把 ra_contrast_beta 0.1→0.0。
# 单条连续从 base 训到 150,每 10 步存档 → 与 FC1 同 recipe/同步数/同 checkpoint 点,可直接叠图。
# 8 卡。约 2-2.5h。用户 2026-07-23 下达:β=0 按 FC1 设置、直跑到 150。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/beta0_matched_fc1_driver.log
log(){ echo "[$(date)] $*" | tee -a "$LOG"; }
step(){ cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }

NAME=Vision-OPD-contrast-uniform-beta0-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-matchfc1
CK=checkpoints/$NAME
DATA=$V/data/virl39k_train_noimg_unfiltered_1img.parquet

# 8 卡必须全空(每卡 <10GB,keep_gpu 的 ~1.7GB baseline 不算占用)
all_free(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1 | awk '{exit !($1<10000)}'; }
until all_free; do log "等 8 卡全空..."; sleep 60; done
log "launching β=0-matched-FC1 (8 GPU, 150 steps, save every 10)"

source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35

MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$NAME \
  ANSWER_VAL_TRAIN_FILE=$DATA \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  actor_rollout_ref.actor.self_distillation.ra_contrast_beta=0.0 \
  trainer.max_actor_ckpt_to_keep=20 \
  data.filter_overlong_prompts=True trainer.total_training_steps=150 \
  > logs/beta0_matched_fc1_150step_$(date +%Y%m%d_%H%M%S).log 2>&1

if [ "$(step "$CK")" -ge 150 ]; then
  for st in 30 60 90 120 150; do
    d="$CK/global_step_${st}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "β=0 step${st} merged"
  done
  log "β=0-matched-FC1 DONE + merged (30/60/90/120/150)"
else
  log "ERROR: β=0 at step $(step "$CK")"
fi
log "=== β=0-matched-FC1 driver finished ==="
