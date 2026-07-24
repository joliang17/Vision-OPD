#!/bin/bash
# FC1 driver（2026-07-18 17:0x，排班表分配 301832790）：
#   2B uniform × unfiltered × 150步 × len6144，save_freq=10 全保留（细曲线，FINAL-WAVE max-step 选择用）
#   等 S2b(GPU1-4) + L1-base(GPU5) + L1-x5(GPU0) 三个 lane 全清后 8 卡 fresh。
#   S1 已按排班移交 301761390（本机 armed driver 已解除）。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/fc1_driver.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
step() { cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
zombie_sweep() { local g p; for g in 0 1 2 3 4 5 6 7; do for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null); do [ "$p" = "5378" ] && continue; log "zombie kill $p GPU$g"; kill -9 "$p" 2>/dev/null || true; done; done; sleep 5; }
all_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1 | awk '{exit !($1<10000)}'; }

S2B=checkpoints/Vision-OPD-baseline-seed1234-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301783374
log "waiting for S2b(>=90) + L1 lanes (log markers)"
while true; do
  ok=1
  [ "$(step "$S2B")" -ge 90 ] || ok=0
  grep -q "L1-base lane exited" logs/s_l_batch.log 2>/dev/null || ok=0
  grep -q "L1-x5 lane exited" logs/s_l_batch.log 2>/dev/null || ok=0
  [ "$ok" -eq 1 ] && break
  sleep 300
done
zombie_sweep
until all_free; do sleep 60; done

FC1=checkpoints/Vision-OPD-contrast-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374
log "launching FC1 (8 GPUs, 150 steps, keep-all)"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-contrast-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374 \
  ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_unfiltered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  trainer.max_actor_ckpt_to_keep=20 \
  data.filter_overlong_prompts=True trainer.total_training_steps=150 \
  > logs/fc1_150step_$(date +%Y%m%d_%H%M%S).log 2>&1
if [ "$(step "$FC1")" -ge 150 ]; then
  for st in 30 60 90 120 150; do
    d="$FC1/global_step_${st}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "FC1 step${st} merged"
  done
  log "FC1 DONE + merged (30/60/90/120/150; 全部 15 档存档保留)"
else
  log "ERROR: FC1 at step $(step "$FC1")"
fi
log "=== FC1 driver finished ==="
