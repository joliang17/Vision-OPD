#!/bin/bash
# trial 301761390 v2 (2026-07-19): attach S1-seed1234 -> L2f (唯一未认领的探针lane) -> FC2
# 撤销 v1 的 L2a-e 段: 301832790 已先认领(其 GPU4-7 lanes), 避免双跑
set -uo pipefail
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=$OPSD/Vision-OPD
cd "$V"
LOG=logs/s1_l2_fc2_driver_301761390.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA_U=$V/data/virl39k_train_noimg_unfiltered_1img.parquet
FAST7=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K

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

log "=== v2: attach S1 -> L2f -> FC2 (L2a-e 让给 301832790) ==="
S1_NAME=Vision-OPD-contrast-uniform-seed1234-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301761390
S1_LOG=$(ls -t logs/s1_seed1234_*.log | head -1)
if wait_train "checkpoints/$S1_NAME" "$S1_LOG" 90; then
  for st in 30 60 90; do
    dd="checkpoints/$S1_NAME/global_step_${st}"
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "S1 step${st} merged"
  done
  log "S1-seed1234 DONE + merged"
else
  log "S1-seed1234 FAILED"
fi
wait_gpus_free

# ---- L2f: 8B ours @8192 (单卡) ----
log "launching L2f (GPU0, 8B ours @8192)"
(
  cd "$OPSD/VLMEvalKit"
  BACKEND=vllm_server PORT=28876 MAX_NEW_TOKENS=8192 \
  EVAL_SETTING_VERSION=qwen3vl_temp0_8192_probe \
  JUDGE_API_NPROC=2 JUDGE_RETRY=12 REQUEST_TIMEOUT=900 RETRY=4 \
  MODEL_PATH=$V/checkpoints/Vision-OPD-contrast-standard-uniformweight-Qwen3-VL-8B-virl39k-UNFILTERED1img-bs32-90step-trial301829143/global_step_90 \
  MODEL_NAME=ours8b_final_eval8192 \
  DATASETS=$FAST7 GPU_IDS=0 bash shell_scripts/eval_model_temp0_4096.sh \
  > $V/logs/l2f_ours8b_eval8192.log 2>&1
  echo "[$(date)] L2f lane exited" >> "$LOG"
) &
L2F_PID=$!

# ---- FC2 与 L2f 并行: FC2 用 GPU1-7? 4B bs32 需8卡吗 -> P27 用8卡; 7卡bs32不整除
# 保守: 等 L2f 完(约2-3h)再8卡跑 FC2
wait $L2F_PID
wait_gpus_free
FC2_NAME=Vision-OPD-contrast-uniform-finecurve-Qwen3-VL-4B-virl39k-UNFILTERED1img-150step-trial301761390
rm -rf "checkpoints/$FC2_NAME"
FC2_LOG=logs/fc2_4b_finecurve_$(date +%Y%m%d_%H%M%S).log
log "launching FC2 (4B 150步 keep-all, len4096 铁律, 8卡)"
MODEL_SIZE=4B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$FC2_NAME ANSWER_VAL_TRAIN_FILE=$DATA_U TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=4096 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  data.filter_overlong_prompts=True trainer.total_training_steps=150 \
  trainer.max_actor_ckpt_to_keep=20 \
  > "$FC2_LOG" 2>&1 &
sleep 60
if wait_train "checkpoints/$FC2_NAME" "$FC2_LOG" 150; then
  for dd in "checkpoints/$FC2_NAME"/global_step_*; do
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "FC2 merged $(basename $dd)"
  done
  log "FC2 DONE: 15档全存不prune (FC批豁免prune政策)"
else
  log "FC2 FAILED"
fi
log "=== v2 driver finished ==="
