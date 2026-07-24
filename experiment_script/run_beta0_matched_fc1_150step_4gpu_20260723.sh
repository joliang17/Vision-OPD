#!/bin/bash
# β=0 崩溃臂,对齐 FC1 设置(只 ra_contrast_beta 0.1→0.0),4卡(GPU0,1,2,7),单条到150,每10步存。
# 全局 batch=32 与 FC1(8卡)一致 → 训练结果等价,仅约2倍慢(~3h)。用户2026-07-23:4卡跑起来。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/beta0_matched_fc1_4gpu_driver.log
log(){ echo "[$(date)] $*" | tee -a "$LOG"; }
step(){ cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
NAME=Vision-OPD-contrast-uniform-beta0-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-matchfc1
CK=checkpoints/$NAME
DATA=$V/data/virl39k_train_noimg_unfiltered_1img.parquet
log "launching β=0-matched-FC1 (4 GPU=0,1,2,7, 150 steps, save every 10)"
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,7 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=$NAME \
  ANSWER_VAL_TRAIN_FILE=$DATA \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  actor_rollout_ref.actor.self_distillation.ra_contrast_beta=0.0 \
  trainer.max_actor_ckpt_to_keep=20 \
  data.filter_overlong_prompts=True trainer.total_training_steps=150 \
  > logs/beta0_matched_fc1_4gpu_$(date +%Y%m%d_%H%M%S).log 2>&1
if [ "$(step "$CK")" -ge 150 ]; then
  for st in 30 60 90 120 150; do
    d="$CK/global_step_${st}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "β=0 step${st} merged"
  done
  log "β=0-matched-FC1 DONE + merged"
else log "ERROR: β=0 at step $(step "$CK")"; fi
log "=== driver finished ==="
