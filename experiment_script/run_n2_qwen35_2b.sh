#!/bin/bash
# N2+扩展（2026-07-18 01:2x，用户下达）：Qwen3.5-2B 下载 → baseline eval ∥ ours 训练
#  1. HF 下载 Qwen/Qwen3.5-2B → cache/Qwen3.5-2B（本机 7890 代理已验证可用）
#  2. baseline eval：GPU0 9-bench（conda shim vllm_server）+ GPU1 ZoomBench canonical
#  3. ours 训练：GPU2-5 contrast-标准 × virl39k-filtered @6144 90步（对齐 T3a 口径，conda env）
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
CACHE=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache
CONDA_BIN=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/envs/qwen35/bin
cd "$V"
LOG=logs/n2_driver.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }

# ---- 1. 下载 ----
M=$CACHE/Qwen3.5-2B
if [ ! -f "$M/config.json" ]; then
  log "downloading Qwen/Qwen3.5-2B via 7890 proxy"
  env -u ALL_PROXY -u all_proxy \
    https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 \
    "$CONDA_BIN/hf" download Qwen/Qwen3.5-2B --local-dir "$M" \
    > logs/n2_download.log 2>&1
  [ -f "$M/config.json" ] || { log "ERROR: 下载失败，见 logs/n2_download.log"; exit 1; }
fi
log "model ready: $M ($(du -sh "$M" | cut -f1))"

# ---- 2. baseline eval（并行两 lane）----
(
  cd "$OPSD/VLMEvalKit"
  PATH=$V/scripts/qwen35_shim:$PATH BACKEND=vllm_server PORT=28850 \
  MODEL_PATH=$M MODEL_NAME=vanilla_qwen35_2b \
  DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
  GPU_IDS=0 bash shell_scripts/eval_model_temp0_4096.sh \
  > $V/logs/n2_base_9bench.log 2>&1
  echo "[$(date)] n2 9-bench lane done" >> $V/$LOG
) &
(
  cd "$V"
  PATH=$V/scripts/qwen35_shim:$PATH bash scripts/run_zoombench_canonical.sh "$M" vanilla_qwen35_2b 1 28851 \
  > $V/logs/n2_base_zoom.log 2>&1
  echo "[$(date)] n2 zoom lane done" >> $V/$LOG
) &

# ---- 3. ours 训练（GPU2-5）----
export PATH="$CONDA_BIN:$PATH"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
T=checkpoints/Vision-OPD-contrast-standard-Qwen3.5-2B-virl39k-filtered-90step-trial301783374
log "launching ours training (GPU2-5, len6144)"
MODEL_PATH=$M CUDA_VISIBLE_DEVICES=2,3,4,5 TRAINER_N_GPUS_PER_NODE=4 \
  EXPERIMENT_NAME=Vision-OPD-contrast-standard-Qwen3.5-2B-virl39k-filtered-90step-trial301783374 \
  ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_filtered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/n2_ours_train.log 2>&1
s=$(cat "$T/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
if [ "${s:-0}" -ge 90 ]; then
  for st in 30 60 90; do
    d="$T/global_step_${st}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "ours step${st} merged"
  done
  log "N2 ours training DONE + merged"
else
  log "ERROR: ours training ended at step ${s}（Qwen3.5-2B 首次走 contrast 路径，若是架构报错需现场排查；OOM 则加 rollout.gpu_memory_utilization=0.55）"
fi
wait
log "=== N2 driver finished ==="
