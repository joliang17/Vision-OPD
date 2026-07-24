#!/bin/bash
# T3b 最终尝试（2026-07-17 01:1x）：等 supervisor v2（P7/P10）结束后，
# rollout池 0.45（0.55时占用已降至117.5GB、差0.6GB；再降0.10≈17.8GB即可覆盖 59.68GB 的重batch backward）
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/t3b_last_attempt.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
CONDA_BIN=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/envs/qwen35/bin
QWEN35_MODEL=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B
T3B=checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301783374
step() { cat "$T3B/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }

log "waiting for supervisor v2 to finish"
while ! grep -q "supervisor v2 finished" logs/supervisor_v2.log 2>/dev/null; do sleep 300; done
while true; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${m:-999999}" -lt 10000 ] && break
  sleep 60
done

log "launching T3b last attempt (from step $(step), rollout pool 0.45)"
export PATH="$CONDA_BIN:$PATH"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
MODEL_PATH="$QWEN35_MODEL" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301783374 \
  ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_filtered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_conservative.sh \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/t3b_pool045_$(date +%Y%m%d_%H%M%S).log 2>&1

if [ "$(step)" -ge 90 ]; then
  for s in 30 60 90; do
    d="$T3B/global_step_${s}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "T3b step${s} merged"
  done
  log "T3b DONE at step 90"
else
  log "ERROR: T3b final attempt still at step $(step) — 建议改用 step80 作为该 run 的终点（merge 60/80 后评测），或换机重试"
fi
