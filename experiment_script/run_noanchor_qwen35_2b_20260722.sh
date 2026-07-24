#!/bin/bash
# no-anchor 跨 scale 之 Qwen3.5-2B：终局配置 + ra_contrast_anchor_coef=0（纯对比比值）
# 口径对齐 N3b(Qwen3.5-2B ours-final)：uniform×unfiltered@**4096**、bs32、90步、8卡、conda qwen35
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/noanchor_qwen35_2b_driver.log
log(){ echo "[$(date)] $*" | tee -a "$LOG"; }
step(){ cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
all_free(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1 | awk '{exit !($1<10000)}'; }

# 串在 uniform β=0 ext200 之后：等它到 200 或其目录不再增长
BETA0=checkpoints/Vision-OPD-contrast-uniform-beta0-Qwen3-VL-2B-virl39k-UNFILTERED1img-ext200-trial301967423
log "armed: 等 uniform β=0 ext200 训完(step>=200)再启动"
while [ "$(step $BETA0)" -lt 200 ]; do sleep 300; done
log "β=0 ext200 已达 200；等 8 卡全空"
until all_free; do sleep 120; done; sleep 60; until all_free; do sleep 60; done

source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

EXP=Vision-OPD-pure-contrast-noanchor-uniform-Qwen3.5-2B-virl39k-UNFILTERED1img-90step-trial301967423
CKPT=checkpoints/$EXP
log "启动 no-anchor Qwen3.5-2B (8卡, len4096, 90步)"
MODEL_PATH=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-2B \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$EXP \
  ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_unfiltered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
  bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=1.0 \
  actor_rollout_ref.actor.self_distillation.ra_contrast_anchor_coef=0.0 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/noanchor_qwen35_2b_$(date +%Y%m%d_%H%M%S).log 2>&1

s=$(step "$CKPT")
if [ "$s" -ge 90 ]; then
  for st in 30 60 90; do d="$CKPT/global_step_${st}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "step${st} merged"; done
  log "no-anchor Qwen3.5-2B DONE + merged"
else log "ERROR: 只到 step $s"; fi
log "=== driver finished ==="
