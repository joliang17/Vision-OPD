#!/bin/bash
# P37a(α=0.75)重跑，串在 P37c(α=1.5)之后（2026-07-21，301832790）：
#   P37a 首跑 step10 OOM(52.74GiB碎片化 transient，b/c 同配置无 OOM)。加 rollout 池 0.55 guard 重跑。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p37a_retry_driver.log
log(){ echo "[$(date)] $*" | tee -a "$LOG"; }
step(){ cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
zombie_sweep(){ local g p; for g in 0 1 2 3 4 5 6 7; do for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null); do [ "$p" = "5378" ] && continue; kill -9 "$p" 2>/dev/null||true; done; done; sleep 5; }
all_free(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1 | awk '{exit !($1<10000)}'; }
C=checkpoints/Vision-OPD-contrast-alpha15-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301832790
log "armed: waiting P37c(alpha15) step90 merged"
while [ ! -f "$C/global_step_90/config.json" ]; do sleep 300; done
log "P37c done; waiting 8 GPUs free"; until all_free; do sleep 120; done; zombie_sweep; until all_free; do sleep 60; done
EXP=Vision-OPD-contrast-alpha075-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301832790
rm -rf "checkpoints/$EXP"   # 清掉 step10 半成品重开
log "relaunch P37a α=0.75 (rollout pool 0.55 guard)"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$EXP ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_unfiltered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=0.75 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/p37a_retry_$(date +%Y%m%d_%H%M%S).log 2>&1
if [ "$(step "checkpoints/$EXP")" -ge 90 ]; then
  for st in 30 60 90; do d="checkpoints/$EXP/global_step_${st}"; [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "P37a-retry step${st} merged"; done
  log "P37a-retry DONE + merged"
else log "ERROR: P37a-retry at step $(step "checkpoints/$EXP")"; fi
log "=== P37a retry driver finished ==="
