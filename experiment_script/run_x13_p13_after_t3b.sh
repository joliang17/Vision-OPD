#!/bin/bash
# X13 + P13 driver（2026-07-17 02:3x，用户分配：P13 本机、P14 归 301761390）
#  等 T3b 最终尝试结束 → 并行：X13 零训练 precheck（GPU7，~30min）+ P13 β=0 训练（GPU0-3，~2.5h）
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/x13_p13_driver.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
BASE2B=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/transformers/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
DATA=$V/data/virl39k_train_noimg_filtered_1img.parquet
P13_DIR=checkpoints/Vision-OPD-contrast-beta0-Qwen3-VL-2B-virl39k-90step-trial301783374

log "waiting for t3b_last_attempt to conclude"
while ! grep -qE "T3b DONE|ERROR" logs/t3b_last_attempt.log 2>/dev/null; do sleep 300; done
while true; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${m:-999999}" -lt 10000 ] && break
  sleep 60
done

log "launching X13 precheck (GPU7) + P13 beta=0 training (GPU0-3)"
CUDA_VISIBLE_DEVICES=7 nohup python3 scripts/precheck_contrast_target_x13.py \
  --teacher-path "$BASE2B" \
  --rollout-dir rollouts/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step \
  --parquet "$DATA" --max-samples 60 --ctrl-mode black --alpha 1.0 --betas "0.1,0" \
  > logs/x13_precheck_$(date +%Y%m%d_%H%M%S).log 2>&1 &
X13_PID=$!

MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-beta0-Qwen3-VL-2B-virl39k-90step-trial301783374 \
  ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_contrast_beta=0.0 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/p13_beta0_2b_virl39k_$(date +%Y%m%d_%H%M%S).log 2>&1 &

while kill -0 "$X13_PID" 2>/dev/null; do sleep 60; done
log "X13 precheck finished; summary: analysis_outputs/contrast_target_precheck_x13/summary.csv"

while [ "$(cat $P13_DIR/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)" -lt 90 ]; do
  if ! ps aux | grep "[m]ain_ppo" | grep -q "contrast-beta0"; then
    sleep 60
    [ "$(cat $P13_DIR/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)" -ge 90 ] && break
    log "ERROR: P13 进程消失于 step $(cat $P13_DIR/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)"; break
  fi
  sleep 300
done
if [ "$(cat $P13_DIR/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)" -ge 90 ]; then
  for s in 30 60 90; do
    d="$P13_DIR/global_step_${s}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "P13 step${s} merged"
  done
  log "P13 DONE + merged"
fi
log "=== x13_p13 driver finished ==="
