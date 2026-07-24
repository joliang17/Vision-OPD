#!/bin/bash
# QS1 retry2 (2026-07-20): 两次@6144 (原始 + rollout池0.55) 都OOM(同59.68GiB签名, step31/50)
# 改用项目已记录的 Qwen3.5 contrast/RA-VAD 安全长度 len4096(与QL1同长度, 但seed=1234不同)
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/qs1_driver_301832756.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA_U=$V/data/virl39k_train_noimg_unfiltered_1img.parquet
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
QWEN35_MODEL=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B

wait_gpus_free() { while true; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "${m:-999999}" -lt 5000 ] && break; sleep 60; done; }
wait_train() {
  local d=$1 f=$2 tgt=$3
  while true; do
    local s; s=$(cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    [ "${s:-0}" -ge "$tgt" ] && return 0
    if [ -f "$f" ]; then
      local age=$(( $(date +%s) - $(stat -c %Y "$f") ))
      [ "$age" -gt 900 ] && ! ps aux | grep -v grep | grep -q verl.trainer.main_ppo && { log "DEAD: $d at step ${s}"; return 1; }
    fi
    sleep 300
  done
}

log "=== QS1 retry2: len4096 (safe recipe, 两次@6144都同签名OOM后改用) ==="
wait_gpus_free
QS1_NAME=Vision-OPD-contrast-uniform-seed1234-Qwen3.5-4B-virl39k-UNFILTERED1img-90step-len4096-trial301832756
rm -rf "checkpoints/$QS1_NAME"
QS1_LOG=logs/qs1_seed1234_len4096_$(date +%Y%m%d_%H%M%S).log
log "launching QS1 @len4096 (8卡): $QS1_NAME"
MODEL_PATH="$QWEN35_MODEL" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$QS1_NAME ANSWER_VAL_TRAIN_FILE=$DATA_U TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  data.seed=1234 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$QS1_LOG" 2>&1 &
sleep 60
if wait_train "checkpoints/$QS1_NAME" "$QS1_LOG" 90; then
  for st in 30 60 90; do
    dd="checkpoints/$QS1_NAME/global_step_${st}"
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "QS1 step${st} merged"
  done
  log "QS1(len4096) DONE + merged"
else
  log "QS1(len4096) FAILED — 看 $QS1_LOG"
fi
log "=== QS1 retry2 driver finished ==="
