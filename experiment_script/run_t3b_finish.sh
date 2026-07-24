#!/bin/bash
# T3b 收尾 driver（2026-07-16 23:5x）：从 step80 resume 到 90 → merge 30/60/90
# 前情：主driver的T3b段因共享conda env被外部动包(tensorboard消失)而失败退出；已修复依赖后单独收尾。
# T6b 不用跑：301761390 已完成并 merge（step90）。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
CONDA_BIN=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/envs/qwen35/bin
QWEN35_MODEL=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B
cd "$V"
LOG=logs/t3b_finish_driver.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
export PATH="$CONDA_BIN:$PATH"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

T3B=checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301783374
log "launching T3b resume (from step $(cat $T3B/latest_checkpointed_iteration.txt))"
MODEL_PATH="$QWEN35_MODEL" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301783374 \
  ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_filtered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_conservative.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/t3b_cons_virl39k_qwen35_301783374_finish.log 2>&1

latest=$(cat "$T3B/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
if [ "${latest:-0}" -ge 90 ] && [ -d "$T3B/global_step_90/actor" ]; then
  for s in 30 60 90; do
    d="$T3B/global_step_${s}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "T3b step${s} merged"
  done
  log "T3b DONE + merged"
else
  log "ERROR: T3b still at step ${latest}, see logs/t3b_cons_virl39k_qwen35_301783374_finish.log"
fi
