#!/bin/bash
# 串行 driver v3（2026-07-18 06:4x）：9-bench(B) 完 → 僵尸扫 → N2-ours 8卡 fresh → FA1 → FA2（均 8 卡）
# N2-ours：step41 的 66.31GB 巨型分配对 4 卡任何池设置都不够（deficit 7GB）——8 卡 per-GPU 负载减半是唯一稳解。
# world_size_4 的 step40 半成品改名弃用（P21 教训：跨卡数不能 resume）。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
CONDA_BIN=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/envs/qwen35/bin
M=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-2B
cd "$V"
LOG=logs/serial_v3.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
step() { cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
merge3() { for st in 30 60 90; do d="$1/global_step_${st}"; [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "$2 step${st} merged"; done; }
zombie_sweep() {
  local g p
  for g in 0 1 2 3 4 5 6 7; do
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null); do
      [ "$p" = "5378" ] && continue   # keep_gpu 绝不能杀
      local c; c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | head -c 60)
      log "zombie sweep: kill $p ($c) on GPU$g"; kill -9 "$p" 2>/dev/null || true
    done
  done
  sleep 5
}
all_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1 | awk '{exit !($1<10000)}'; }

log "waiting for vanilla 9-bench (B lane) to finish"
while ! grep -q "B DONE" logs/recovery_20260718.log 2>/dev/null; do
  ps -p 2392810 >/dev/null 2>&1 || break
  sleep 300
done
zombie_sweep
until all_free; do sleep 60; done

# ---- N2-ours 8卡 fresh ----
N2T=checkpoints/Vision-OPD-contrast-standard-Qwen3.5-2B-virl39k-filtered-90step-trial301783374
[ -d "$N2T" ] && [ "$(step "$N2T")" -lt 90 ] && { mv "$N2T" "${N2T}-4gpu-partial-DISCARD"; log "renamed 4-gpu partial (step40) aside"; }
export PATH="$CONDA_BIN:$PATH"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
log "launching N2-ours fresh on 8 GPUs"
MODEL_PATH=$M CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-contrast-standard-Qwen3.5-2B-virl39k-filtered-90step-trial301783374 \
  ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_filtered_1img.parquet \
  TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/n2_ours_8gpu_$(date +%Y%m%d_%H%M%S).log 2>&1
if [ "$(step "$N2T")" -ge 90 ]; then merge3 "$N2T" "N2ours"; log "N2 ours DONE (8gpu fresh)"; else log "ERROR: N2 ours 8gpu at step $(step "$N2T")"; fi
zombie_sweep; until all_free; do sleep 60; done

# ---- FA1 / FA2 8卡 fresh ----
run_fa8() {
  local suf=$1 a=$2 tag=$3
  local name="Vision-OPD-contrast-uniform-${suf}-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301783374"
  local dir="checkpoints/$name"
  [ -d "$dir" ] && [ "$(step "$dir")" -lt 90 ] && { rm -rf "$dir"; log "cleared 4-gpu partial $name"; }
  log "launching $tag fresh on 8 GPUs"
  MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
    EXPERIMENT_NAME=$name ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_unfiltered_1img.parquet \
    TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
    bash scripts/run_experiment_contrast_standard.sh \
    actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
    actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=$a \
    data.filter_overlong_prompts=True trainer.total_training_steps=90 \
    > logs/${tag}_8gpu_$(date +%Y%m%d_%H%M%S).log 2>&1
  if [ "$(step "$dir")" -ge 90 ]; then merge3 "$dir" "$tag"; log "$tag DONE"; else log "ERROR: $tag at step $(step "$dir")"; fi
  zombie_sweep; until all_free; do sleep 60; done
}
run_fa8 "alpha05" 0.5 "fa1_uniform_alpha05_unfiltered"
run_fa8 "alpha20" 2.0 "fa2_uniform_alpha20_unfiltered"
log "=== serial v3 finished ==="
