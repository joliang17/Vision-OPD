#!/bin/bash
# β=0 续训 100→120（Table2 崩溃曲线补 step120，devbox 07-23 下达）
# 复用 ext100 dir（已含 step90/100，world_size_4，conda qwen35 存的）：
#   latest=100 ⇒ resume 100 → 存 110、120。同 torch 版本 resume，无 DeviceMesh 跨版本坑。
# 分片 world_size_4 ⇒ 必须 4 卡。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/beta0_ext120_driver.log
log(){ echo "[$(date)] $*" | tee -a "$LOG"; }
NAME=Vision-OPD-contrast-beta0-Qwen3-VL-2B-virl39k-ext100-trial301967423
CKPT=checkpoints/$NAME
DATA=$V/data/virl39k_train_noimg_filtered_1img.parquet

s0=$(cat $CKPT/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
[ "$s0" -ne 100 ] && { log "FATAL: 期望 step100，实际 $s0，停止"; exit 1; }
log "resume $NAME 100->120 on GPU0-3 (world_size_4)"

source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35

MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=$NAME ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_contrast_beta=0.0 \
  data.filter_overlong_prompts=True trainer.total_training_steps=120 \
  > logs/beta0_ext120_$(date +%Y%m%d_%H%M%S).log 2>&1

s=$(cat $CKPT/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
if [ "$s" -ge 120 ]; then
  d=$CKPT/global_step_120
  [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "step120 merged"
  log "β=0 ext120 DONE (latest=$s)"
else log "ERROR: 只到 step $s"; fi
log "=== beta0 ext120 driver finished ==="
