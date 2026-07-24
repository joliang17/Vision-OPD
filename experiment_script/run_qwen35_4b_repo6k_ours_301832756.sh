#!/bin/bash
# 2026-07-22（用户下达）：Qwen3.5-4B "ours"(contrast_standard) 在原生 visionopd 6k 数据集
# (data/train.parquet, 6241条) 上训练，而不是本项目主口径的 virl39k。
#  关键点：不要设 ANSWER_VAL_TRAIN_FILE（那是 N4/N6 用来把训练语料"偷梁换柱"成 virl39k 的机制，
#  见 run_vision_opd_ra_vad.sh:99-165 —— TASK_TRAIN_FILE 默认就是 train.parquet，只要不覆盖
#  ANSWER_VAL_TRAIN_FILE，脚本会用 prepare_answer_val_split.py 自动切分 train.parquet 出训练/验证集，
#  这就是原生 repo-6k 语料）。
# len 沿用 QS1/N4 教训直接从 4096 起步，避免 Qwen3.5 系在 6144 的 seed 依赖型 OOM。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/qwen35_4b_repo6k_driver_301832756.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
QWEN35_MODEL=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B

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

log "=== Qwen3.5-4B ours(contrast_standard) x repo-6k(train.parquet) @len4096, 62步 ==="
wait_gpus_free
NAME=Vision-OPD-contrast-standard-Qwen3.5-4B-repo6k-62step-trial301832756
rm -rf "checkpoints/$NAME"
RUN_LOG=logs/qwen35_4b_repo6k_ours_$(date +%Y%m%d_%H%M%S).log
log "launching (8卡): $NAME"
MODEL_PATH="$QWEN35_MODEL" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$NAME TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  data.filter_overlong_prompts=True trainer.total_training_steps=62 \
  > "$RUN_LOG" 2>&1 &
sleep 60
if wait_train "checkpoints/$NAME" "$RUN_LOG" 62; then
  for st in 20 40 62; do
    dd="checkpoints/$NAME/global_step_${st}"
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "step${st} merged"
  done
  log "Qwen3.5-4B repo6k ours DONE + merged"
else
  log "FAILED — 看 $RUN_LOG（若 OOM，同 QS1 套路：先降rollout gpu_memory_utilization，再降len）"
fi
log "=== driver finished ==="
