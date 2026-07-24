#!/bin/bash
# P29 批（2026-07-17 18:0x，用户分配本机）：4 个消融转 unfiltered 口径，2B × unfiltered virl39k 90步
#   lane A (GPU0-3, 等 P21 完): α=0.5 → α=2.0
#   lane B (GPU4-7, 等 P24 完): no-sample-gate → no-eos-exempt
# 配方 = P9（unfiltered parquet, len6144, bs32, 90步），只加各自 override
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/p29_batch_driver.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA=$V/data/virl39k_train_noimg_unfiltered_1img.parquet
step() { cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
# ⚠️ 存活检测必须用 /proc/*/cmdline：ps aux 和 pgrep -f 对 verl 的 ~8KB 命令行都会截断漏匹配
#（07-17 已三次因此误判：driver 干等9h、supervisor 误报、本 driver 首版把 p29 提前砸进在跑的 P21/P24）
proc_alive() {  # <pattern>
  local pat=$1 p
  for p in $(pgrep -f main_ppo); do
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "$pat" && return 0
  done
  return 1
}
merge90() { for s in 30 60 90; do d="$1/global_step_${s}"; [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "$2 step${s} merged"; done; }
gpus_free() { local ids=$1; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$ids" | sort -n | tail -1 | awk '{exit !($1<10000)}'; }

run_one() { # <gpus> <name-suffix> <override> <logtag>
  local gpus=$1 suf=$2 ovr=$3 tag=$4
  local name="Vision-OPD-contrast-${suf}-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301783374"
  local dir="checkpoints/$name"
  until gpus_free "$gpus"; do log "$tag: GPU$gpus 未空，续等"; sleep 120; done
  log "launching $tag on GPU$gpus"
  MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=$gpus TRAINER_N_GPUS_PER_NODE=4 \
    EXPERIMENT_NAME=$name ANSWER_VAL_TRAIN_FILE=$DATA TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
    bash scripts/run_experiment_contrast_standard.sh \
    $ovr data.filter_overlong_prompts=True trainer.total_training_steps=90 \
    > logs/${tag}_$(date +%Y%m%d_%H%M%S).log 2>&1
  if [ "$(step "$dir")" -ge 90 ]; then merge90 "$dir" "$tag"; log "$tag DONE"; else log "ERROR: $tag ended at step $(step "$dir")"; fi
}

laneA() {
  log "laneA waiting for P21 (GPU0-3)"
  while proc_alive "150step-trial301829143"; do sleep 300; done
  until gpus_free "0,1,2,3"; do sleep 60; done
  run_one "0,1,2,3" "alpha05-nogate" "actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=0.5" "p29a_alpha05_unfiltered"
  run_one "0,1,2,3" "alpha20-nogate" "actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=2.0" "p29c_alpha20_unfiltered"
}
laneB() {
  log "laneB waiting for P24 (GPU4-7)"
  while proc_alive "baseline-Qwen3-VL-4B-virl39k-UNFILTERED"; do sleep 300; done
  until gpus_free "4,5,6,7"; do sleep 60; done
  run_one "4,5,6,7" "standard-nosamplegate" "actor_rollout_ref.actor.self_distillation.ra_no_sample_gate=True" "p29b_nosamplegate_unfiltered"
  run_one "4,5,6,7" "noeosexempt" "actor_rollout_ref.actor.self_distillation.ra_contrast_exclude_token_ids=[]" "p29d_noeosexempt_unfiltered"
}
laneA & A=$!
laneB & B=$!
wait $A $B
log "=== P29 batch driver finished ==="
