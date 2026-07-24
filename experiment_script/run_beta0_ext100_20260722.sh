#!/bin/bash
# β=0 续训 90→100（Table2 崩溃曲线补 step100，devbox 07-22 下达）
# 坑规避：① trial301783374 的 step90 已被 prune，只能用 ext200 的 step90
#         ② 不能直接用 ext200 dir（latest=110 会从110续），故拷进独立 -ext100 dir 并置 latest=90
#         ③ 分片 world_size_4 ⇒ 必须 4 卡
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/beta0_ext100_driver.log
log(){ echo "[$(date)] $*" | tee -a "$LOG"; }
NAME=Vision-OPD-contrast-beta0-Qwen3-VL-2B-virl39k-ext100-trial301967423
CKPT=checkpoints/$NAME
DATA=$V/data/virl39k_train_noimg_filtered_1img.parquet

s0=$(cat $CKPT/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
[ "$s0" -ne 90 ] && { log "FATAL: 期望 step90，实际 $s0，停止"; exit 1; }
log "resume $NAME 90->100 on GPU0-3 (world_size_4)"

source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35

MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=$NAME ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_contrast_beta=0.0 \
  data.filter_overlong_prompts=True trainer.total_training_steps=100 \
  > logs/beta0_ext100_$(date +%Y%m%d_%H%M%S).log 2>&1

s=$(cat $CKPT/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
if [ "$s" -ge 100 ]; then
  d=$CKPT/global_step_100
  [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "step100 merged"
  log "β=0 ext100 DONE (latest=$s)"
else log "ERROR: 只到 step $s"; fi
log "=== beta0 ext100 driver finished ==="
