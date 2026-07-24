#!/bin/bash
# "ours" (contrast-sharpened distillation target) ported to the CROSS-MODEL OPD setting:
#   teacher = frozen Qwen3-VL-8B-Instruct, student = Qwen3-VL-2B-Instruct.
#
# Why this script exists
# ---------------------
# The two capabilities had never been combined before: the *contrast* target
# (ra_target_mode=contrast, i.e. the current "ours") lives only in Vision-OPD, while the
# *fixed cross-model teacher* (teacher_model_source=fixed) was only ever driven from the
# Vision-OPD-ra-vad-4b2b-noimg fork, whose ra_vad.py predates contrast. So every previous
# 8B->2B run used either plain reverse-KL OPD or the superseded `vaopd_grouped` RA
# weighting — never ours. This script is ours-final, with the teacher swapped for the 8B.
#
# Everything below is byte-identical in 口径 to the ours-final 2B row (uniform x unfiltered
# @4096, bs32, 90 steps, lr 2e-6, ctrl = black image, alpha=1.0, beta=0.1, no gate,
# EOS/pad tilt-exempt) EXCEPT the teacher:
#   self-distillation ours : teacher_hi = EMA(student)@real-img , ctrl = EMA(student)@black
#   THIS script (OPD ours) : teacher_hi = frozen 8B @real-img   , ctrl = frozen 8B @black
#
# The 8B and 2B checkpoints share vocab (151936) and vision patch/merge/temporal config
# (16/2/2), so the student processor's tokenisation is valid for the teacher forward.
set -uo pipefail

V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35

STUDENT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/transformers/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
TEACHER=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b

export HF_HOME=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

TOTAL_STEPS="${TOTAL_STEPS:-90}"
EXP="${EXPERIMENT_NAME:-Vision-OPD-OPDours-8Bteacher-Qwen3-VL-2B-virl39k-UNFILTERED1img-${TOTAL_STEPS}step-$(hostname | tail -c 10)}"
CKPT="checkpoints/$EXP"
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/opd_ours_8b2b_${TS}.log"
mkdir -p logs

echo "[$(date)] launching $EXP  (8 GPU, bs32, len4096, ${TOTAL_STEPS} steps, teacher=8B fixed)" | tee -a "$LOG"

MODEL_PATH="$STUDENT" MODEL_SIZE=2B \
TEACHER_MODEL_SOURCE=fixed TEACHER_MODEL_PATH="$TEACHER" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
EXPERIMENT_NAME="$EXP" \
ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_unfiltered_1img.parquet \
TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
  bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  data.filter_overlong_prompts=True \
  trainer.total_training_steps="$TOTAL_STEPS" \
  "$@" >> "$LOG" 2>&1
RC=$?
echo "[$(date)] training exited rc=$RC" | tee -a "$LOG"

s=$(cat "$CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
echo "[$(date)] reached step $s" | tee -a "$LOG"
if [ "$s" -ge "$TOTAL_STEPS" ]; then
    for st in 30 60 90; do
        d="$CKPT/global_step_${st}"
        [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && \
            bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && echo "[$(date)] step${st} merged" | tee -a "$LOG"
    done
    echo "[$(date)] DONE $EXP" | tee -a "$LOG"
else
    echo "[$(date)] INCOMPLETE (rc=$RC, step=$s) — see $LOG" | tee -a "$LOG"
fi
