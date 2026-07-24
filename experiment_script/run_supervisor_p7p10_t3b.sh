#!/bin/bash
# 总收尾 supervisor（2026-07-17 00:3x）：
#  1. 等 P7/P10（已在跑）到 step90 → merge 30/60/90
#  2. GPU 空后重试 T3b 最后3步（step80→90），加显存杠杆：rollout池0.55 + allocator GC
#     （上次死在 step88 backward 差 0.53GB；keep_gpu 的 1.69GB 绝不能动）
#  3. T3b 到 90 → merge 30/60/90
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/supervisor_p7p10_t3b.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
QWEN35_MODEL=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B
CONDA_BIN=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/envs/qwen35/bin

P7=checkpoints/Vision-OPD-contrast-alpha05-nogate-Qwen3-VL-2B-virl39k-90step-trial301783374
P10=checkpoints/Vision-OPD-contrast-standard-nosamplegate-Qwen3-VL-2B-virl39k-90step-trial301783374
T3B=checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301783374

step() { cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
merge90() { # <dir> <name>
  for s in 30 60 90; do
    d="$1/global_step_${s}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "$2 step${s} merged"
  done
}

# ---- 1. 等 P7/P10 ----
for pair in "$P7:P7:alpha05-nogate" "$P10:P10:nosamplegate"; do
  d=$(echo "$pair" | cut -d: -f1); n=$(echo "$pair" | cut -d: -f2); pat=$(echo "$pair" | cut -d: -f3)
  while [ "$(step "$d")" -lt 90 ]; do
    if ! pgrep -f "$pat" >/dev/null 2>&1; then
      sleep 60
      [ "$(step "$d")" -ge 90 ] && break
      log "ERROR: $n 进程消失且只到 step $(step "$d")"; break
    fi
    sleep 300
  done
  [ "$(step "$d")" -ge 90 ] && { merge90 "$d" "$n"; log "$n DONE"; }
done

# ---- 2. T3b 最后3步 ----
if [ "$(step "$T3B")" -lt 90 ]; then
  while true; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${m:-999999}" -lt 10000 ] && break
    sleep 60
  done
  log "launching T3b final steps (from step $(step "$T3B")) with memory levers"
  export PATH="$CONDA_BIN:$PATH"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
  PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6" \
  MODEL_PATH="$QWEN35_MODEL" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
    EXPERIMENT_NAME=Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301783374 \
    ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_filtered_1img.parquet \
    TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
    bash scripts/run_experiment_contrast_conservative.sh \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    data.filter_overlong_prompts=True trainer.total_training_steps=90 \
    > logs/t3b_final3_$(date +%Y%m%d_%H%M%S).log 2>&1
  if [ "$(step "$T3B")" -ge 90 ]; then merge90 "$T3B" "T3b"; log "T3b DONE"; else log "ERROR: T3b still at step $(step "$T3B")"; fi
fi
log "=== supervisor finished ==="
