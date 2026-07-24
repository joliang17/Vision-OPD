#!/bin/bash
# FA1/FA2 driver（2026-07-18 04:0x，队列分配 301832790）：
#   终局配置消融 = uniform（ra_uniform_weight=True）× unfiltered × 2B × 90步，单因子改 α
#   FA1: α=0.5 → FA2: α=2.0，串行，等 N2 的 ours 训练（GPU2-5）结束后启动
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/fa1_fa2_driver.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA=$V/data/virl39k_train_noimg_unfiltered_1img.parquet
step() { cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
proc_alive() { local p; for p in $(pgrep -f main_ppo); do tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "$1" && return 0; done; return 1; }
gpus_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | sort -n | tail -1 | awk '{exit !($1<10000)}'; }
merge90() { for s in 30 60 90; do d="$1/global_step_${s}"; [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "$2 step${s} merged"; done; }

run_fa() { # <suffix> <alpha> <tag>
  local suf=$1 a=$2 tag=$3
  local name="Vision-OPD-contrast-uniform-${suf}-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301783374"
  local dir="checkpoints/$name"
  until gpus_free "2,3,4,5"; do sleep 120; done
  log "launching $tag (GPU2-5, alpha=$a)"
  MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=2,3,4,5 TRAINER_N_GPUS_PER_NODE=4 \
    EXPERIMENT_NAME=$name ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
    bash scripts/run_experiment_contrast_standard.sh \
    actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
    actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=$a \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    data.filter_overlong_prompts=True trainer.total_training_steps=90 \
    > logs/${tag}_$(date +%Y%m%d_%H%M%S).log 2>&1
  if [ "$(step "$dir")" -ge 90 ]; then merge90 "$dir" "$tag"; log "$tag DONE"; else log "ERROR: $tag ended at step $(step "$dir")（unfiltered 重batch OOM 则带 rollout.gpu_memory_utilization=0.55 resume）"; fi
}

log "FA v2: resume-aware, pool 0.55 baked in"
run_fa "alpha05" 0.5 "fa1_uniform_alpha05_unfiltered"
run_fa "alpha20" 2.0 "fa2_uniform_alpha20_unfiltered"
log "=== FA1/FA2 driver finished ==="
