#!/bin/bash
# trial 301832756: QS1 v2 — 用共享 NAS conda qwen35 env（不是 pip user-local 装法）
# 上一次失败是因为新设备没有 Qwen3.5 支持的 python 栈；miniconda3 整个装在 NAS 共享盘上，
# qwen35 env 对所有机器天然可用，不需要重装。conda activate 后直接跑训练脚本即可
# （run_vision_opd_ra_vad.sh 本身不感知 conda，靠当前 shell 的 PATH 决定用哪个 python）。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/qs1_driver_301832756.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA_U=$V/data/virl39k_train_noimg_unfiltered_1img.parquet

source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
QWEN35_MODEL=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B
python -c "import torch,vllm,transformers; from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration; print('conda qwen35 env OK: torch',torch.__version__,'vllm',vllm.__version__,'transformers',transformers.__version__)" | tee -a "$LOG"

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

log "=== QS1 v2 (conda qwen35): Qwen3.5-4B uniform x unfiltered @6144 seed1234 ==="
wait_gpus_free
QS1_NAME=Vision-OPD-contrast-uniform-seed1234-Qwen3.5-4B-virl39k-UNFILTERED1img-90step-trial301832756
rm -rf "checkpoints/$QS1_NAME"
QS1_LOG=logs/qs1_seed1234_conda_$(date +%Y%m%d_%H%M%S).log
log "launching QS1 (8卡, conda qwen35): $QS1_NAME"
MODEL_PATH="$QWEN35_MODEL" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$QS1_NAME ANSWER_VAL_TRAIN_FILE=$DATA_U TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  data.seed=1234 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$QS1_LOG" 2>&1 &
sleep 60
if wait_train "checkpoints/$QS1_NAME" "$QS1_LOG" 90; then
  for st in 30 60 90; do
    dd="checkpoints/$QS1_NAME/global_step_${st}"
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "QS1 step${st} merged"
  done
  log "QS1 DONE + merged"
else
  log "QS1 FAILED — 看 $QS1_LOG"
fi
log "=== QS1 v2 driver finished ==="
