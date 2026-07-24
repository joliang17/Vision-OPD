#!/bin/bash
# N2 ours resume（2026-07-18 05:2x）：step40 OOM 后带池 0.55 续跑（world_size_4 → 用 GPU0,1,6,7，
# 等 GPU0 的 vanilla 9-bench 评测结束）。FA1/FA2 在 GPU2-5 并行不受影响。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
CONDA_BIN=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/envs/qwen35/bin
M=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-2B
cd "$V"
LOG=logs/n2_ours_resume_driver.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
T=checkpoints/Vision-OPD-contrast-standard-Qwen3.5-2B-virl39k-filtered-90step-trial301783374
step() { cat "$T/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }

log "waiting for GPU0/1/6/7 to be free (vanilla 9-bench on GPU0)"
while true; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,6,7 | sort -n | tail -1)
  [ "${m:-999999}" -lt 10000 ] && break
  sleep 180
done
export PATH="$CONDA_BIN:$PATH"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
log "resuming N2 ours from step $(step) on GPU0,1,6,7 (pool 0.55)"
MODEL_PATH=$M CUDA_VISIBLE_DEVICES=0,1,6,7 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-standard-Qwen3.5-2B-virl39k-filtered-90step-trial301783374 \
  ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_filtered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/n2_ours_resume_$(date +%Y%m%d_%H%M%S).log 2>&1
if [ "$(step)" -ge 90 ]; then
  for st in 30 60 90; do
    d="$T/global_step_${st}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "ours step${st} merged"
  done
  log "N2 ours DONE + merged"
else
  log "ERROR: N2 ours still at step $(step)（再 OOM 就降池 0.45，仍不过则 len 降 4096 并记录口径）"
fi
