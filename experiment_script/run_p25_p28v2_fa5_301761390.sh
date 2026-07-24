#!/bin/bash
# trial 301761390 driver v2 (2026-07-18, FINAL-WAVE 改型后重排):
#   [attach] P25 = Qwen3.5 answer-hint × unfiltered 90步 (已在跑, 8卡)
#   P28v2   = Qwen3.5 uniform × unfiltered **直接 total_training_steps=150**
#             (FINAL-WAVE 要 step90/120/150 三点; P28 未开跑, 一次训到150比"先90再续训"干净,
#              完全绕开跨机/续训 FSDP resume 的 DTensor 坑; save_freq=10 白拿全部点)
#   之后并行: FA5 = uniform×unfiltered×2B×90步 + data.seed=1234 (GPU0-3, 终局配置 variance 点, fresh-run only)
#           + no-think baseline eval (GPU4, ENABLE_THINKING=false, Qwen3.5 base)
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p25_p28_driver_301761390.log   # 沿用原driver日志
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
QWEN35_MODEL=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B
DATA_UNF=$V/data/virl39k_train_noimg_unfiltered_1img.parquet

wait_gpus_free() {
  while true; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${m:-999999}" -lt 10000 ] && break
    sleep 60
  done
}
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
merge_steps() { # $1=dir $2=空格分隔的步数列表
  local d=$1; shift
  for st in "$@"; do
    local dd="$d/global_step_${st}"
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && \
      bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "$(basename "$d") step${st} merged"
  done
}

log "=== driver v2: attach P25 -> P28@150 -> FA5 + nothink-eval ==="
P25_NAME=Vision-OPD-baseline-Qwen3.5-4B-virl39k-UNFILTERED1img-90step-trial301761390
P25_LOG=$(ls -t logs/p25_answerhint_qwen35_unfiltered_*.log | head -1)
log "attaching P25 ($P25_LOG)"
if wait_train "checkpoints/$P25_NAME" "$P25_LOG" 90; then
  merge_steps "checkpoints/$P25_NAME" 30 60 90; log "P25 DONE + merged"
else
  log "P25 FAILED — 看 $P25_LOG, 继续 P28"
fi
wait_gpus_free

P28_NAME=Vision-OPD-contrast-standard-uniformweight-Qwen3.5-4B-virl39k-UNFILTERED1img-150step-trial301761390
P28_LOG=logs/p28v2_uniformweight_qwen35_unfiltered_150_$(date +%Y%m%d_%H%M%S).log
log "launching P28v2 (150步一次跑齐): $P28_NAME"
env PYTHONNOUSERSITE=0 MODEL_PATH="$QWEN35_MODEL" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$P28_NAME ANSWER_VAL_TRAIN_FILE=$DATA_UNF TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  data.filter_overlong_prompts=True trainer.total_training_steps=150 \
  > "$P28_LOG" 2>&1 &
sleep 60
if wait_train "checkpoints/$P28_NAME" "$P28_LOG" 150; then
  merge_steps "checkpoints/$P28_NAME" 30 60 90 120 150; log "P28v2 DONE + merged (90/120/150 全齐)"
else
  log "P28v2 FAILED — 看 $P28_LOG"
fi
wait_gpus_free

# ---- FA5 (GPU0-3) + no-think eval (GPU4) 并行 ----
FA5_NAME=Vision-OPD-contrast-uniformweight-seed1234-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301761390
FA5_LOG=logs/fa5_uniform_seed1234_$(date +%Y%m%d_%H%M%S).log
log "launching FA5 (GPU0-3, fresh-run only 勿resume) + nothink-eval (GPU4)"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=$FA5_NAME ANSWER_VAL_TRAIN_FILE=$DATA_UNF TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  data.seed=1234 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$FA5_LOG" 2>&1 &
GPU_IDS=4 PORT=8312 nohup bash experiment_script/eval_qwen35_base_nothink.sh > logs/nothink_base_eval.log 2>&1 &
sleep 60
wait_train "checkpoints/$FA5_NAME" "$FA5_LOG" 90 && { merge_steps "checkpoints/$FA5_NAME" 30 60 90; log "FA5 DONE + merged"; } || log "FA5 FAILED (fresh-run only: 修好后 rm -rf 重跑)"
log "=== driver v2 finished (nothink eval 独立收尾, 看 logs/nothink_base_eval.log) ==="
