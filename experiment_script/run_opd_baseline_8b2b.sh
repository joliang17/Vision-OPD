#!/bin/bash
# OPD BASELINE, 8B teacher -> 2B student. VLM port of thunlp/OPD's `token_reward_direct`
# ("Rethinking On-Policy Distillation of LLMs", arXiv:2604.13016), only_stu + student_p.
#
# What makes this different from every previous 8B->2B run in this project
# -----------------------------------------------------------------------
# Previous runs backpropagated the teacher/student divergence as a *differentiable*
# distribution-matching loss (full-vocab reverse KL, or the contrast target). That is
# full-strength distribution matching, and it drags the 8B's verbose answer STYLE into the
# 2B along with its capability -- the OPD-ours run collapsed V* 72.77 -> 60.73 with <1%
# truncation, i.e. genuinely worse, not an eval artifact.
#
# thunlp's objective instead DETACHES the divergence into a per-(token, candidate) reward
# and feeds it to the standard PPO clipped policy loss as a 3-D advantage:
#
#     kl_val  = S_logp - T_on_S            # (B, L, K), K = the STUDENT's own top-K
#     weights = softmax_K(S_logp)          # renormalised student prob inside top-K
#     adv     = -(kl_val * weights)        # detached  -> token_reward_direct
#     loss    = PPO clipped surrogate over the top-K log-probs
#
# so the only gradient path is grad-log-pi, bounded by ratio clipping + dual clip. The
# implementation is verified bit-identical to thunlp's reference formula (diff 0.00e+00).
#
# Deviations from thunlp's LLM script, and why:
#   - rollout n = 8 (their 4): matches this project's "ours" runs so the objective is the
#     only difference when comparing against the 2B ours row.
#   - lr = 1e-6 (their value) rather than this project's 2e-6: 1e-6 is what they validated
#     for THIS objective. Override with LR= if you want the ours-matched 2e-6.
#   - data/len/bs follow the project's ours 口径 (virl39k unfiltered 1img @4096, bs32,
#     90 steps) so the result drops straight into the main table.
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
OPD_TOPK="${OPD_TOPK:-16}"
EXP="${EXPERIMENT_NAME:-Vision-OPD-OPDbaseline-tokenreward-8Bteacher-Qwen3-VL-2B-virl39k-UNFILTERED1img-${TOTAL_STEPS}step}"
CKPT="checkpoints/$EXP"
LOG="logs/opd_baseline_8b2b_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "[$(date)] launching $EXP (8 GPU, bs32, len4096, ${TOTAL_STEPS} steps, topk=${OPD_TOPK}, teacher=8B fixed)" | tee -a "$LOG"

EXPERIMENT=opd \
MODEL_PATH="$STUDENT" MODEL_SIZE=2B \
TEACHER_MODEL_SOURCE=fixed TEACHER_MODEL_PATH="$TEACHER" \
OPD_TOPK="$OPD_TOPK" OPD_REWARD_WEIGHT_MODE="${OPD_REWARD_WEIGHT_MODE:-student_p}" \
LR="${LR:-1e-6}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-8}" \
EXPERIMENT_NAME="$EXP" \
ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_unfiltered_1img.parquet \
TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
  bash scripts/run_vision_opd_ra_vad.sh \
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
    # merge under conda (transformers 5.5) omits the 4.57-style processor files vLLM needs
    python3 - "$CKPT" "$STUDENT" <<'PY' | tee -a "$LOG"
import json, os, shutil, sys
ckpt, base = sys.argv[1], sys.argv[2]
for st in (30, 60, 90):
    d = os.path.join(ckpt, f"global_step_{st}")
    if not os.path.isdir(d):
        continue
    tc = os.path.join(d, "tokenizer_config.json")
    if os.path.isfile(tc):
        cfg = json.load(open(tc))
        if isinstance(cfg.get("extra_special_tokens"), list):
            cfg.pop("extra_special_tokens")            # list-typed field breaks vLLM's tokenizer load
            json.dump(cfg, open(tc, "w"), ensure_ascii=False, indent=2)
    for f in ("merges.txt", "vocab.json", "preprocessor_config.json", "video_preprocessor_config.json"):
        dst = os.path.join(d, f)
        if not os.path.exists(dst) and os.path.exists(os.path.join(base, f)):
            shutil.copy2(os.path.join(base, f), dst)
    print(f"[postfix] step{st} ok")
PY
    echo "[$(date)] DONE $EXP" | tee -a "$LOG"
else
    echo "[$(date)] INCOMPLETE (rc=$RC, step=$s) — see $LOG" | tee -a "$LOG"
fi
