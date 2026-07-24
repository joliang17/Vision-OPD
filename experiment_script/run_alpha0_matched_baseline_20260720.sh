#!/bin/bash
# α=0 matched baseline（2026-07-20，用户 paper 请求）：
#   终局 ours-uniform 行的逐项完全 matched 对照，唯一变量 = contrast tilt 项 α 从 1.0 → 0.0。
#   保留：ra_uniform_weight=True / β=0.1 plausibility mask / EOS-豁免 exclude_token_ids /
#         forward-KL(divergence_alpha=0) / EMA teacher / unfiltered virl39k / 90步 / len6144 / 8卡。
#   α=0 → build_contrast_target 退化为 log_softmax(lp_hi 限制在 plausibility set)，即纯温度软化
#   EMA-teacher 自蒸馏（无对比锐化）。validator 已放宽允许 α=0（actor.py，α<0 仍拒）。
#   判定：ours-uniform(主表highlight行) vs 本行 —— 差值即"对比项本身"的净贡献（隔离 reweight/自蒸馏）。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/alpha0_matched_driver.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
step() { cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
zombie_sweep() { local g p; for g in 0 1 2 3 4 5 6 7; do for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null); do [ "$p" = "5378" ] && continue; log "zombie kill $p GPU$g"; kill -9 "$p" 2>/dev/null || true; done; done; sleep 5; }
all_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1 | awk '{exit !($1<10000)}'; }

log "armed: waiting for all 8 GPUs to free (current G1 9-bench + D-group evals)"
until all_free; do sleep 120; done
log "all GPUs report <10GB; zombie sweep then launch"
zombie_sweep
until all_free; do sleep 60; done

EXP=Vision-OPD-contrast-alpha0-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301832790
CKPT=checkpoints/$EXP
log "launching α=0 matched baseline (8 GPUs, 90 steps)"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$EXP \
  ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_unfiltered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=0.0 \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/alpha0_matched_90step_$(date +%Y%m%d_%H%M%S).log 2>&1

if [ "$(step "$CKPT")" -ge 90 ]; then
  for st in 30 60 90; do
    d="$CKPT/global_step_${st}"
    [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "α0 step${st} merged"
  done
  log "α0 baseline DONE + merged (30/60/90)"
else
  log "ERROR: α0 baseline at step $(step "$CKPT")"
fi
log "=== α0 matched baseline driver finished ==="
