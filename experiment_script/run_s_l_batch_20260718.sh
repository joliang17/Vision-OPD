#!/bin/bash
# S/L 批 driver（2026-07-18 16:2x，301832790 认领 S1/S2b/L1）：
#  now:  S2b = answer-hint × unfiltered × seed1234 fresh（GPU1-4，top-k蒸馏无OOM压力）
#  now:  L1-base = 4B base 7-bench @ MAX_NEW_TOKENS=8192（GPU5）
#  tail3(GPU0) 完 → L1-x5 = X5 ckpt(len6144训) 7-bench @8192（GPU0）
#  全清 → S1 = uniform × unfiltered × seed1234 fresh 8卡（主配置 seed 加固；8卡=unfiltered 唯一稳解）
# 完成判定一律用 checkpoint/产物文件，不用进程模式。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "$V"
LOG=logs/s_l_batch.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
step() { cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
merge3() { for st in 30 60 90; do d="$1/global_step_${st}"; [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "$2 step${st} merged"; done; }
zombie_sweep() { local g p; for g in 0 1 2 3 4 5 6 7; do for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null); do [ "$p" = "5378" ] && continue; log "zombie kill $p GPU$g"; kill -9 "$p" 2>/dev/null || true; done; done; sleep 5; }
gpus_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | sort -n | tail -1 | awk '{exit !($1<10000)}'; }
DATA_U=$V/data/virl39k_train_noimg_unfiltered_1img.parquet
Q3VL4B=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
FAST7=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K

S2B=checkpoints/Vision-OPD-baseline-seed1234-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301783374
S1=checkpoints/Vision-OPD-contrast-uniform-seed1234-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301783374
VAN=$OPSD/VLMEvalKit/outputs_vllm_curated/vanilla_qwen35_2b_qwen3vl2b_temp0_4096_generic_eval/normal_scoring/vanilla_qwen35_2b_HallusionBench_normal_score.csv

# ---- S2b（GPU1-4）----
(
  log "S2b: launching answer-hint x unfiltered seed1234 (GPU1-4)"
  MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=1,2,3,4 TRAINER_N_GPUS_PER_NODE=4 \
    EXPERIMENT_NAME=Vision-OPD-baseline-seed1234-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301783374 \
    ANSWER_VAL_TRAIN_FILE=$DATA_U TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
    bash scripts/run_experiment_baseline.sh \
    data.seed=1234 data.filter_overlong_prompts=True trainer.total_training_steps=90 \
    > logs/s2b_answerhint_seed1234_$(date +%Y%m%d_%H%M%S).log 2>&1
  if [ "$(step "$S2B")" -ge 90 ]; then merge3 "$S2B" "S2b"; log "S2b DONE"; else log "ERROR: S2b at step $(step "$S2B")"; fi
) &
PS2B=$!

# ---- L1-base（GPU5，立即）----
(
  cd "$OPSD/VLMEvalKit"
  BACKEND=vllm_server PORT=28860 MAX_NEW_TOKENS=8192 \
  EVAL_SETTING_VERSION=qwen3vl_temp0_8192_probe \
  MODEL_PATH=$Q3VL4B MODEL_NAME=vanilla_qwen3vl4b_eval8192 \
  DATASETS=$FAST7 GPU_IDS=5 bash shell_scripts/eval_model_temp0_4096.sh \
  > $V/logs/l1_base8192.log 2>&1
  log "L1-base lane exited"
) &
PL1B=$!

# ---- L1-x5（GPU0，等 tail3 出齐三文件）----
(
  D=$OPSD/VLMEvalKit/outputs_vllm_curated/vanilla_qwen35_2b_qwen3vl2b_temp0_4096_generic_eval/normal_scoring
  while ! { [ -f "$D/vanilla_qwen35_2b_HallusionBench_normal_score.csv" ] && [ -f "$D/vanilla_qwen35_2b_POPE_normal_score.csv" ]; }; do sleep 300; done
  until gpus_free "0"; do sleep 120; done
  cd "$OPSD/VLMEvalKit"
  BACKEND=vllm_server PORT=28861 MAX_NEW_TOKENS=8192 \
  EVAL_SETTING_VERSION=qwen3vl_temp0_8192_probe \
  MODEL_PATH=$V/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-4B-virl39k-filtered-90step-trial301761390/global_step_90 \
  MODEL_NAME=x5_ours6144_eval8192 \
  DATASETS=$FAST7 GPU_IDS=0 bash shell_scripts/eval_model_temp0_4096.sh \
  > $V/logs/l1_x5_8192.log 2>&1
  log "L1-x5 lane exited"
) &
PL1X=$!

wait $PS2B $PL1B $PL1X
zombie_sweep
until gpus_free "0,1,2,3,4,5,6,7"; do sleep 60; done

# ---- S1 8卡 fresh ----
log "S1: launching uniform x unfiltered seed1234 fresh (8 GPUs)"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=Vision-OPD-contrast-uniform-seed1234-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301783374 \
  ANSWER_VAL_TRAIN_FILE=$DATA_U TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  data.seed=1234 data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > logs/s1_uniform_seed1234_$(date +%Y%m%d_%H%M%S).log 2>&1
if [ "$(step "$S1")" -ge 90 ]; then merge3 "$S1" "S1"; log "S1 DONE"; else log "ERROR: S1 at step $(step "$S1")"; fi
log "=== S/L batch finished ==="
