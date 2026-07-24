#!/bin/bash
# trial 301761390: paper 主表/消融补跑链（2026-07-17 认领）
#   P5  = 4B contrast-标准 × virl39k-filtered 90步 (主表 4B ours 行)      8卡, len6144 OOM则降4096
#   P6  = 4B answer-hint(baseline) × virl39k-filtered 90步 (主表 4B OPSD) 8卡, 同上
#   P7  = 2B α=0.5 无gate × virl39k 90步 (α消融)   GPU0-3  ← 从301783374接手(其env缺tensorboard死于step0)
#   P10 = 2B no_sample_gate × virl39k 90步 (w_t分解) GPU4-7 ← 同上
# 成功判定一律看 latest_checkpointed_iteration.txt >= 90，liveness 看训练日志 mtime（不 pgrep 实验名）。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p5p6p7p10_driver_301761390.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA=$V/data/virl39k_train_noimg_filtered_1img.parquet
TRIAL=trial301761390

wait_gpus_free() {
  while true; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${m:-999999}" -lt 10000 ] && break
    sleep 60
  done
}

# 等训练结束: $1=ckpt目录 $2=训练日志 $3=目标step。返回0=达标
wait_train() {
  local d=$1 f=$2 tgt=$3
  while true; do
    local s; s=$(cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    [ "${s:-0}" -ge "$tgt" ] && return 0
    # liveness: 日志 15 分钟没动 && 无 main_ppo 进程 => 死了
    if [ -f "$f" ]; then
      local age=$(( $(date +%s) - $(stat -c %Y "$f") ))
      if [ "$age" -gt 900 ] && ! pgrep -f verl.trainer.main_ppo >/dev/null 2>&1; then
        log "DEAD: $d at step ${s} (log idle ${age}s, no main_ppo)"
        return 1
      fi
    fi
    sleep 300
  done
}

merge_steps() {
  local d=$1
  for st in 30 60 90; do
    local dd="$d/global_step_${st}"
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && \
      bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "$(basename "$d") step${st} merged"
  done
}

# 4B 训练带 OOM 降级重试: $1=名字前缀 $2=入口脚本 [额外 hydra 参数...]
run_4b() {
  local name=$1 entry=$2; shift 2
  for len in 6144 4096; do
    local ckpt=checkpoints/$name
    local tlog=logs/${name}_len${len}_$(date +%Y%m%d_%H%M%S).log
    log "launching $name len=$len entry=$entry"
    MODEL_SIZE=4B TRAINER_N_GPUS_PER_NODE=8 EXPERIMENT_NAME=$name \
      ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=$len \
      nohup bash scripts/$entry "$@" \
      data.filter_overlong_prompts=True trainer.total_training_steps=90 \
      > "$tlog" 2>&1 &
    sleep 60
    if wait_train "$ckpt" "$tlog" 90; then
      merge_steps "$ckpt"; log "$name DONE + merged (len=$len)"; return 0
    fi
    pkill -f verl.trainer.main_ppo; pkill -f "ray::"; sleep 30; wait_gpus_free
    if grep -qE "CUDA out of memory|OutOfMemoryError" "$tlog"; then
      log "$name OOM at len=$len, 清目录后降级重试"
      rm -rf "$ckpt"
    else
      log "$name FAILED (非OOM), 不重试, 看 $tlog"; return 1
    fi
  done
  log "$name FAILED even at len=4096"; return 1
}

log "=== P5/P6/P7/P10 chain start ($TRIAL) ==="
wait_gpus_free

# ---- P5: 4B contrast-标准 (主表 ours) ----
run_4b "Vision-OPD-contrast-standard-Qwen3-VL-4B-virl39k-filtered-90step-$TRIAL" \
  run_experiment_contrast_standard.sh
P5_RC=$?
wait_gpus_free

# ---- P6: 4B answer-hint (主表 OPSD) ----
run_4b "Vision-OPD-baseline-Qwen3-VL-4B-virl39k-filtered-90step-$TRIAL" \
  run_experiment_baseline.sh
P6_RC=$?
wait_gpus_free

# ---- P7 + P10: 2B 消融, 4+4 并行 ----
P7_NAME=Vision-OPD-contrast-alpha05-nogate-Qwen3-VL-2B-virl39k-90step-$TRIAL
P10_NAME=Vision-OPD-contrast-standard-nosamplegate-Qwen3-VL-2B-virl39k-90step-$TRIAL
P7_LOG=logs/p7_alpha05_nogate_2b_${TRIAL}_$(date +%Y%m%d_%H%M%S).log
P10_LOG=logs/p10_nosamplegate_2b_${TRIAL}_$(date +%Y%m%d_%H%M%S).log
log "launching P7 (GPU0-3) + P10 (GPU4-7)"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=$P7_NAME ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=0.5 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$P7_LOG" 2>&1 &
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=4,5,6,7 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=$P10_NAME ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_no_sample_gate=True \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$P10_LOG" 2>&1 &
sleep 60
wait_train "checkpoints/$P7_NAME" "$P7_LOG" 90 && { merge_steps "checkpoints/$P7_NAME"; log "P7 DONE + merged"; } || log "P7 FAILED"
wait_train "checkpoints/$P10_NAME" "$P10_LOG" 90 && { merge_steps "checkpoints/$P10_NAME"; log "P10 DONE + merged"; } || log "P10 FAILED"

log "=== chain finished: P5_RC=$P5_RC P6_RC=$P6_RC ==="
