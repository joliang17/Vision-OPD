#!/bin/bash
# trial 301761390: P5/P6 链结束后接跑 P14 + P8（2026-07-17 02:3x 认领）
#   P14 = 无 EOS tilt 豁免（exclude_token_ids=[]），2B × virl39k 90步   GPU0-3  ← 用户指定本机
#   P8  = α=2.0 无gate，2B × virl39k 90步（α曲线第三点）               GPU4-7  ← 抢跑（301829143 链尾有双跑保护会自动跳过）
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p14_p8_driver_301761390.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA=$V/data/virl39k_train_noimg_filtered_1img.parquet
TRIAL=trial301761390

wait_train() { # $1=ckpt目录 $2=日志 $3=目标step
  local d=$1 f=$2 tgt=$3
  while true; do
    local s; s=$(cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    [ "${s:-0}" -ge "$tgt" ] && return 0
    if [ -f "$f" ]; then
      local age=$(( $(date +%s) - $(stat -c %Y "$f") ))
      # ps aux 而非 pgrep -f: verl cmdline 超 4KB 会被 pgrep 漏掉（301832790 的教训）
      [ "$age" -gt 900 ] && ! ps aux | grep -v grep | grep -q verl.trainer.main_ppo && { log "DEAD: $d at step ${s}"; return 1; }
    fi
    sleep 300
  done
}
merge_steps() {
  for st in 30 60 90; do
    local dd="$1/global_step_${st}"
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && \
      bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "$(basename "$1") step${st} merged"
  done
}

log "=== waiting for P5/P6 resume driver to finish (pid via kill -0 + log 标记双条件) ==="
RESUME_PID=$(ps aux | grep -v grep | grep run_p5_p6_only_301761390.sh | awk '{print $2}' | head -1)
log "resume driver pid=${RESUME_PID:-unknown}"
while true; do
  # 结束标记出现 = 正常结束（权威信号，进程探测只作 fallback）
  grep -q "resume driver finished" logs/p5p6p7p10_driver_301761390.log && break
  if [ -n "${RESUME_PID:-}" ]; then
    kill -0 "$RESUME_PID" 2>/dev/null || { sleep 60; grep -q "resume driver finished" logs/p5p6p7p10_driver_301761390.log || log "WARN: resume driver(pid $RESUME_PID) 死了但没写结束标记, 请人工核对 P6"; break; }
  fi
  sleep 300
done
while true; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${m:-999999}" -lt 10000 ] && break
  sleep 60
done

# P8 双跑保护：别的 trial 已有进度就只跑 P14
RUN_P8=1
for d in checkpoints/Vision-OPD-contrast-alpha20-nogate-Qwen3-VL-2B-virl39k-90step-trial*; do
  [ -d "$d" ] || continue
  case "$d" in *"$TRIAL") continue;; esac
  s=$(cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  [ "${s:-0}" -gt 0 ] && { log "P8 已被 $(basename "$d") 跑出 step $s, 本机跳过 P8"; RUN_P8=0; }
done

P14_NAME=Vision-OPD-contrast-noeosexempt-Qwen3-VL-2B-virl39k-90step-$TRIAL
P8_NAME=Vision-OPD-contrast-alpha20-nogate-Qwen3-VL-2B-virl39k-90step-$TRIAL
P14_LOG=logs/p14_noeosexempt_$(date +%Y%m%d_%H%M%S).log
P8_LOG=logs/p8_alpha20_nogate_$(date +%Y%m%d_%H%M%S).log

log "launching P14 (GPU0-3)"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=$P14_NAME ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_contrast_exclude_token_ids=[] \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$P14_LOG" 2>&1 &

if [ "$RUN_P8" = 1 ]; then
  log "launching P8 (GPU4-7)"
  MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=4,5,6,7 TRAINER_N_GPUS_PER_NODE=4 \
    EXPERIMENT_NAME=$P8_NAME ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
    nohup bash scripts/run_experiment_contrast_standard.sh \
    actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=2.0 \
    data.filter_overlong_prompts=True trainer.total_training_steps=90 \
    > "$P8_LOG" 2>&1 &
fi
sleep 60
wait_train "checkpoints/$P14_NAME" "$P14_LOG" 90 && { merge_steps "checkpoints/$P14_NAME"; log "P14 DONE + merged"; } || log "P14 FAILED — 看 $P14_LOG"
if [ "$RUN_P8" = 1 ]; then
  wait_train "checkpoints/$P8_NAME" "$P8_LOG" 90 && { merge_steps "checkpoints/$P8_NAME"; log "P8 DONE + merged"; } || log "P8 FAILED — 看 $P8_LOG"
fi
log "=== P14/P8 driver finished ==="
