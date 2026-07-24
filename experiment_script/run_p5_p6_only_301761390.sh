#!/bin/bash
# trial 301761390: P5→P6 resume driver（2026-07-17 00:5x 替换 run_p5_p6_p7_p10_301761390.sh —
# P7/P10 段切给 301829143 跑, 本机只留 4B 两行, 防双跑）。
# P5 训练已在跑（00:56 启动, len6144）; 本脚本 attach 上去等它, OOM 则降 4096 重跑, 然后接 P6。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p5p6p7p10_driver_301761390.log   # 沿用原driver日志, 历史连续
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
wait_train() {
  local d=$1 f=$2 tgt=$3
  while true; do
    local s; s=$(cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    [ "${s:-0}" -ge "$tgt" ] && return 0
    if [ -f "$f" ]; then
      local age=$(( $(date +%s) - $(stat -c %Y "$f") ))
      [ "$age" -gt 900 ] && ! pgrep -f verl.trainer.main_ppo >/dev/null 2>&1 && { log "DEAD: $d at step ${s}"; return 1; }
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
# $1=名字 $2=入口 $3=attach日志(空=从头跑) ; OOM 降级 6144→4096
run_4b() {
  local name=$1 entry=$2 attach=${3:-}
  for len in 6144 4096; do
    local ckpt=checkpoints/$name
    local tlog
    if [ -n "$attach" ] && [ "$len" = 6144 ]; then
      tlog=$attach; log "attaching to already-running $name len=6144 ($tlog)"
    else
      tlog=logs/${name}_len${len}_$(date +%Y%m%d_%H%M%S).log
      log "launching $name len=$len entry=$entry"
      MODEL_SIZE=4B TRAINER_N_GPUS_PER_NODE=8 EXPERIMENT_NAME=$name \
        ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=$len \
        nohup bash scripts/$entry \
        data.filter_overlong_prompts=True trainer.total_training_steps=90 \
        > "$tlog" 2>&1 &
      sleep 60
    fi
    if wait_train "$ckpt" "$tlog" 90; then
      merge_steps "$ckpt"; log "$name DONE + merged (len=$len)"; return 0
    fi
    pkill -f verl.trainer.main_ppo; pkill -f "ray::"; sleep 30; wait_gpus_free
    if grep -qE "CUDA out of memory|OutOfMemoryError" "$tlog"; then
      log "$name OOM at len=$len, 清目录后降级重试"; rm -rf "$ckpt"
    else
      log "$name FAILED (非OOM), 不重试, 看 $tlog"; return 1
    fi
  done
  log "$name FAILED even at len=4096"; return 1
}

log "=== resume driver: P5(attach) -> P6, P7/P10 已移交 301829143 ==="
P5_LOG=$(ls -t logs/Vision-OPD-contrast-standard-Qwen3-VL-4B-virl39k-filtered-90step-${TRIAL}_len6144_*.log | head -1)
run_4b "Vision-OPD-contrast-standard-Qwen3-VL-4B-virl39k-filtered-90step-$TRIAL" \
  run_experiment_contrast_standard.sh "$P5_LOG"
P5_RC=$?
wait_gpus_free
run_4b "Vision-OPD-baseline-Qwen3-VL-4B-virl39k-filtered-90step-$TRIAL" \
  run_experiment_baseline.sh
P6_RC=$?
log "=== resume driver finished: P5_RC=$P5_RC P6_RC=$P6_RC ==="
