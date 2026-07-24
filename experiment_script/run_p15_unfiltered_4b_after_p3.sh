#!/bin/bash
# P15 driver（2026-07-17 06:0x，用户排队）：P3(150步边界扫描)完成后，
# contrast-标准 × UNFILTERED virl39k(36,039单图) × Qwen3-VL-4B，90步，8卡。
# 与 P9(2B UNFILTERED)/P5(4B filtered) 构成 filter敏感性 × scale 的 2×2。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p15_driver.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
QWEN3VL4B=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
P3=checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-virl39k-150step-trial301829143
P15=checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-4B-virl39k-UNFILTERED1img-90step-trial301783374
step() { cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }

log "waiting for P3 to reach 150"
while [ "$(step "$P3")" -lt 150 ]; do
  grep -q "ERROR: P3" logs/p3_resume_driver.log 2>/dev/null && { log "ERROR: P3 driver 报错，P15 暂停等人工处理"; exit 1; }
  sleep 300
done
while true; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${m:-999999}" -lt 10000 ] && break
  sleep 60
done

log "launching P15 (8 GPUs)"
MODEL_PATH="$QWEN3VL4B" MODEL_SIZE=4B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-contrast-standard-Qwen3-VL-4B-virl39k-UNFILTERED1img-90step-trial301783374 \
  ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_unfiltered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/p15_unfiltered_4b_$(date +%Y%m%d_%H%M%S).log 2>&1

if [ "$(step "$P15")" -ge 90 ]; then
  for s in 30 60 90; do
    d="$P15/global_step_${s}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "P15 step${s} merged"
  done
  log "P15 DONE + merged"
else
  log "ERROR: P15 ended at step $(step "$P15")（若 OOM：先试 rollout.gpu_memory_utilization=0.55，再 0.45——T3b 验证过的路径）"
fi
