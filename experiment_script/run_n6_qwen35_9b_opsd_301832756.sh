#!/bin/bash
# N6（2026-07-21，用户拍板，最高优）：Qwen3.5-9B OPSD/answer-hint x unfiltered 训练，补齐9B组OPSD行
#  同 P25(Qwen3.5-4B OPSD) 配方换 9B，用 run_experiment_baseline.sh（S2c 2B OPSD 同款launcher）。
#  len 直接从 4096 起（同 N4 的教训：9B 全词表蒸馏显存更紧，不重演4B那次 @6144 OOM 弯路）。
#  GPU 线（本脚本）与 judge 重判线（CPU/API）并行跑，互不冲突。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
CACHE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache
CONDA_BIN=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/envs/qwen35/bin
cd "$V"
LOG=logs/n6_driver_301832756.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA_U=$V/data/virl39k_train_noimg_unfiltered_1img.parquet

M=$CACHE/Qwen3.5-9B
if [ ! -f "$M/config.json" ]; then
  log "downloading Qwen/Qwen3.5-9B via 7890 proxy"
  env -u ALL_PROXY -u all_proxy \
    https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 \
    "$CONDA_BIN/hf" download Qwen/Qwen3.5-9B --local-dir "$M" \
    > logs/n6_download.log 2>&1
  [ -f "$M/config.json" ] || { log "ERROR: 下载失败，见 logs/n6_download.log"; exit 1; }
fi
log "model ready: $M ($(du -sh "$M" | cut -f1))"

source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35

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

log "=== N6: Qwen3.5-9B OPSD(answer-hint) x unfiltered @len4096, 90步 ==="
wait_gpus_free
N6_NAME=Vision-OPD-baseline-Qwen3.5-9B-virl39k-UNFILTERED1img-90step-trial301832756
rm -rf "checkpoints/$N6_NAME"
N6_LOG=logs/n6_opsd_$(date +%Y%m%d_%H%M%S).log
log "launching N6 (8卡): $N6_NAME"
MODEL_PATH="$M" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$N6_NAME ANSWER_VAL_TRAIN_FILE=$DATA_U TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
  nohup bash scripts/run_experiment_baseline.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$N6_LOG" 2>&1 &
sleep 60
if wait_train "checkpoints/$N6_NAME" "$N6_LOG" 90; then
  for st in 30 60 90; do
    dd="checkpoints/$N6_NAME/global_step_${st}"
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "N6 step${st} merged"
  done
  log "N6 DONE + merged"
else
  log "N6 FAILED — 看 $N6_LOG（若 OOM，降 rollout gpu_memory_utilization 或再降 len，同 QS1 套路）"
fi
log "=== N6 driver finished ==="
