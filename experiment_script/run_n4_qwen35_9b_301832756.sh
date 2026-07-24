#!/bin/bash
# N4（2026-07-20，用户下达）：Qwen3.5-9B 加入底座矩阵，主配置(uniform×unfiltered×alpha=1.0默认)训练
#  1. HF 下载 Qwen/Qwen3.5-9B -> cache/Qwen3.5-9B（本机 7890 代理已验证）
#  2. ours 训练：uniform × unfiltered，90步，conda qwen35 env，8卡
#     len 直接用 4096（不是6144）：QS1 已实证 Qwen3.5-4B @6144 seed依赖型OOM(59.68GiB签名)，
#     9B 比 4B 更大、全词表蒸馏显存更紧，直接从安全长度起步，不重演 4B 那次 OOM->retry->retry 的弯路
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
CACHE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache
CONDA_BIN=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/envs/qwen35/bin
cd "$V"
LOG=logs/n4_driver_301832756.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA_U=$V/data/virl39k_train_noimg_unfiltered_1img.parquet

# ---- 1. 下载 ----
M=$CACHE/Qwen3.5-9B
if [ ! -f "$M/config.json" ]; then
  log "downloading Qwen/Qwen3.5-9B via 7890 proxy"
  env -u ALL_PROXY -u all_proxy \
    https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 \
    "$CONDA_BIN/hf" download Qwen/Qwen3.5-9B --local-dir "$M" \
    > logs/n4_download.log 2>&1
  [ -f "$M/config.json" ] || { log "ERROR: 下载失败，见 logs/n4_download.log"; exit 1; }
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

log "=== N4: Qwen3.5-9B uniform x unfiltered @len4096, 90步 ==="
wait_gpus_free
N4_NAME=Vision-OPD-contrast-standard-uniformweight-Qwen3.5-9B-virl39k-UNFILTERED1img-90step-trial301832756
rm -rf "checkpoints/$N4_NAME"
N4_LOG=logs/n4_ours_$(date +%Y%m%d_%H%M%S).log
log "launching N4 (8卡): $N4_NAME"
MODEL_PATH="$M" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$N4_NAME ANSWER_VAL_TRAIN_FILE=$DATA_U TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$N4_LOG" 2>&1 &
sleep 60
if wait_train "checkpoints/$N4_NAME" "$N4_LOG" 90; then
  for st in 30 60 90; do
    dd="checkpoints/$N4_NAME/global_step_${st}"
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "N4 step${st} merged"
  done
  log "N4 DONE + merged"
else
  log "N4 FAILED — 看 $N4_LOG（若 OOM，下一步降 rollout gpu_memory_utilization 或再降 len）"
fi
log "=== N4 driver finished ==="
