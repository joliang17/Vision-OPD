#!/bin/bash
# trial 301761390 (2026-07-19): S1-seed1234 (移交回本机) -> L2 六路 @8192 长度探针 -> FC2 (4B 细曲线, 低优垫底)
set -uo pipefail
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=$OPSD/Vision-OPD
cd "$V"
LOG=logs/s1_l2_fc2_driver_301761390.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
DATA_U=$V/data/virl39k_train_noimg_unfiltered_1img.parquet
FAST7=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K
Q3VL2B=$(ls -d /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/*/ | head -1)
QWEN35=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B

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

log "=== S1-seed1234 -> L2 batch -> FC2 ==="
wait_gpus_free

# ---- S1-seed1234 (8卡) ----
S1_NAME=Vision-OPD-contrast-uniform-seed1234-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301761390
rm -rf "checkpoints/$S1_NAME" "checkpoints/Vision-OPD-contrast-uniformweight-seed1234-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301761390"
S1_LOG=logs/s1_seed1234_$(date +%Y%m%d_%H%M%S).log
log "launching S1-seed1234 (8卡)"
MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAINER_N_GPUS_PER_NODE=8 \
  EXPERIMENT_NAME=$S1_NAME ANSWER_VAL_TRAIN_FILE=$DATA_U TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
  nohup bash scripts/run_experiment_contrast_standard.sh \
  actor_rollout_ref.actor.self_distillation.ra_uniform_weight=True \
  data.seed=1234 data.filter_overlong_prompts=True trainer.total_training_steps=90 \
  > "$S1_LOG" 2>&1 &
sleep 60
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

# ---- L2 六路并行 @8192 (JUDGE_API_NPROC=2 防止合打429) ----
l2lane() { # $1=gpu $2=port $3=model_path $4=model_name $5=extra_env(shim或空)
  local gpu=$1 port=$2 mp=$3 mn=$4 shim=${5:-}
  (
    cd "$OPSD/VLMEvalKit"
    env ${shim:+PATH=$V/scripts/qwen35_shim:$PATH} \
    BACKEND=vllm_server PORT=$port MAX_NEW_TOKENS=8192 \
    EVAL_SETTING_VERSION=qwen3vl_temp0_8192_probe \
    JUDGE_API_NPROC=2 JUDGE_RETRY=12 REQUEST_TIMEOUT=900 RETRY=4 \
    MODEL_PATH=$mp MODEL_NAME=$mn \
    DATASETS=$FAST7 GPU_IDS=$gpu bash shell_scripts/eval_model_temp0_4096.sh \
    > $V/logs/l2_${mn}.log 2>&1
    echo "[$(date)] L2 lane $mn exited" >> "$LOG"
  ) &
}
log "launching L2 lanes (6路 @8192)"
l2lane 0 28870 "$Q3VL2B" vanilla_qwen3vl2b_eval8192
l2lane 1 28871 "$V/checkpoints/Vision-OPD-contrast-standard-uniformweight-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301829143/global_step_90" ours2b_final_eval8192
l2lane 2 28872 "$V/checkpoints/Vision-OPD-baseline-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301761390/global_step_90" opsd2b_unf_eval8192
l2lane 3 28873 "$V/checkpoints/Vision-OPD-contrast-standard-uniformweight-Qwen3-VL-4B-virl39k-UNFILTERED1img-90step-trial301829143/global_step_90" ours4b_final_eval8192
l2lane 4 28874 "$QWEN35" vanilla_qwen35_4b_eval8192 shim
l2lane 5 28875 "$V/checkpoints/Vision-OPD-contrast-standard-uniformweight-Qwen3.5-4B-virl39k-UNFILTERED1img-150step-trial301761390/global_step_90" ours_qwen35_final_eval8192 shim
l2lane 6 28876 "$V/checkpoints/Vision-OPD-contrast-standard-uniformweight-Qwen3-VL-8B-virl39k-UNFILTERED1img-bs32-90step-trial301829143/global_step_90" ours8b_final_eval8192
wait
log "L2 batch all lanes exited"
wait_gpus_free

# ---- FC2: 4B uniform×unfiltered 150步 len4096 keep-all 不prune ----
FC2_NAME=Vision-OPD-contrast-uniform-finecurve-Qwen3-VL-4B-virl39k-UNFILTERED1img-150step-trial301761390
rm -rf "checkpoints/$FC2_NAME"
FC2_LOG=logs/fc2_4b_finecurve_$(date +%Y%m%d_%H%M%S).log
log "launching FC2 (4B 150步 keep-all; len4096 铁律)"
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
log "=== S1/L2/FC2 driver finished ==="
