#!/bin/bash
# S2c retry3 (2026-07-20): 系统栈三连坏(flash_attn ABI / 缺tensorboard / numpy-pandas ABI)——
# 停止逐层打补丁，改用已验证干净的 conda qwen35 env（此前证实能正常训练 Qwen3-VL，不只 Qwen3.5）。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/s2c_driver_301832756.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA_U=$V/data/virl39k_train_noimg_unfiltered_1img.parquet

source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
python -c "import torch,transformers,tensorboard; from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration; print('conda qwen35 OK for Qwen3-VL:', torch.__version__, transformers.__version__)" | tee -a "$LOG"

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

log "=== S2c retry3 (conda qwen35): 2B answer-hint x unfiltered, seed=777 ==="
wait_gpus_free
S2C_NAME=Vision-OPD-baseline-seed777-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301832756
rm -rf "checkpoints/$S2C_NAME"
S2C_LOG=logs/s2c_seed777_conda_$(date +%Y%m%d_%H%M%S).log
log "launching S2c retry3 (8卡, conda qwen35): $S2C_NAME"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$S2C_NAME ANSWER_VAL_TRAIN_FILE=$DATA_U TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_baseline.sh \
  data.seed=777 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$S2C_LOG" 2>&1 &
sleep 60
if wait_train "checkpoints/$S2C_NAME" "$S2C_LOG" 90; then
  for st in 30 60 90; do
    dd="checkpoints/$S2C_NAME/global_step_${st}"
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "S2c step${st} merged"
  done
  log "S2c DONE + merged"
else
  log "S2c FAILED again (conda attempt) — 看 $S2C_LOG"
fi
log "=== S2c retry3 (conda) driver finished ==="
