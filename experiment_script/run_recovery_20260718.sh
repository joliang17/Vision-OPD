#!/bin/bash
# 恢复总driver（2026-07-18 06:2x）：去共居串行化
#  A. N2-ours 独占续跑（GPU0,1,6,7, world_size_4, pool 0.45）——与 B 并行但不同卡组
#  B. vanilla_qwen35_2b 9-bench 重跑（GPU2，deps 已修）
#  C. A+B 全清后：FA1 → FA2 8 卡 fresh（丢弃 4 卡的 step10 半成品，per-GPU 负载减半绕开重batch OOM）
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
CONDA_BIN=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/envs/qwen35/bin
M=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-2B
cd "$V"
LOG=logs/recovery_20260718.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
step() { cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
merge3() { for st in 30 60 90; do d="$1/global_step_${st}"; [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "$2 step${st} merged"; done; }
DATA_U=$V/data/virl39k_train_noimg_unfiltered_1img.parquet
N2T=checkpoints/Vision-OPD-contrast-standard-Qwen3.5-2B-virl39k-filtered-90step-trial301783374

# ---- A: N2 ours 独占续跑 ----
(
  export PATH="$CONDA_BIN:$PATH"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
  log "A: resuming N2 ours solo (GPU0,1,6,7, pool 0.45, from step $(step $N2T))"
  MODEL_PATH=$M CUDA_VISIBLE_DEVICES=0,1,6,7 TRAINER_N_GPUS_PER_NODE=4 \
    EXPERIMENT_NAME=Vision-OPD-contrast-standard-Qwen3.5-2B-virl39k-filtered-90step-trial301783374 \
    ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_filtered_1img.parquet \
    TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
    bash scripts/run_experiment_contrast_standard.sh \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    data.filter_overlong_prompts=True trainer.total_training_steps=90 \
    > logs/n2_ours_solo_$(date +%Y%m%d_%H%M%S).log 2>&1
  if [ "$(step $N2T)" -ge 90 ]; then merge3 "$N2T" "N2ours"; log "A DONE: N2 ours merged"; else log "A ERROR: N2 ours at step $(step $N2T)"; fi
) &
A=$!

# ---- B: 9-bench on GPU2 ----
(
  cd "$OPSD/VLMEvalKit"
  PATH=$V/scripts/qwen35_shim:$PATH BACKEND=vllm_server PORT=28853 \
  MODEL_PATH=$M MODEL_NAME=vanilla_qwen35_2b \
  DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
  GPU_IDS=2 bash shell_scripts/eval_model_temp0_4096.sh > $V/logs/n2_base_9bench_rerun.log 2>&1
  log "B DONE: vanilla_qwen35_2b 9-bench lane exited"
) &
B=$!
wait $A $B

# ---- C: FA1/FA2 8卡 fresh ----
run_fa8() { # <suffix> <alpha> <tag>
  local suf=$1 a=$2 tag=$3
  local name="Vision-OPD-contrast-uniform-${suf}-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301783374"
  local dir="checkpoints/$name"
  [ -d "$dir" ] && [ "$(step "$dir")" -lt 90 ] && { log "C: clearing partial world_size_4 dir $name (step $(step "$dir"))"; rm -rf "$dir"; }
  while true; do
    mfree=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${mfree:-999999}" -lt 10000 ] && break
    sleep 120
  done
  log "C: launching $tag fresh on 8 GPUs"
  MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
    EXPERIMENT_NAME=$name ANSWER_VAL_TRAIN_FILE=$DATA_U TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
    bash scripts/run_experiment_contrast_standard.sh \
    actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
    actor_rollout_ref.actor.self_distillation.ra_contrast_alpha=$a \
    data.filter_overlong_prompts=True trainer.total_training_steps=90 \
    > logs/${tag}_8gpu_$(date +%Y%m%d_%H%M%S).log 2>&1
  if [ "$(step "$dir")" -ge 90 ]; then merge3 "$dir" "$tag"; log "C: $tag DONE"; else log "C ERROR: $tag at step $(step "$dir")"; fi
}
run_fa8 "alpha05" 0.5 "fa1_uniform_alpha05_unfiltered"
run_fa8 "alpha20" 2.0 "fa2_uniform_alpha20_unfiltered"
log "=== recovery driver finished ==="
