#!/bin/bash
# FC4 + L2 批（2026-07-19 02:0x，301832790 认领）：
#   GPU0-3: FC4 = 2B answer-hint × unfiltered × 150步 keep-all（P22 配方；与 FC1 组双线细曲线）
#   GPU4-7: L2a/b/c/d 四路 @8192 长度探针（7-bench 快组）
#   a/b 清场后 GPU4/5 接 L2e 两臂（Qwen3.5 base+ours, shim）
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "$V"
LOG=logs/fc4_l2_driver.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
step() { cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
FAST7=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K
PROBE=qwen3vl_temp0_8192_probe
Q2B=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/transformers/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
Q35_4B=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B

eval8192() { # <gpu> <port> <model_path> <name> [shim]
  local gpu=$1 port=$2 mp=$3 name=$4 shim=${5:-}
  cd "$OPSD/VLMEvalKit"
  local pathprefix=""
  [ "$shim" = "shim" ] && export PATH="$V/scripts/qwen35_shim:$PATH"
  BACKEND=vllm_server PORT=$port MAX_NEW_TOKENS=8192 EVAL_SETTING_VERSION=$PROBE \
  MODEL_PATH="$mp" MODEL_NAME="$name" DATASETS=$FAST7 GPU_IDS=$gpu \
  bash shell_scripts/eval_model_temp0_4096.sh > "$V/logs/${name}.log" 2>&1
  cd "$V"; log "$name lane exited"
}

# ---- FC4 (GPU0-3) ----
(
  FC4=checkpoints/Vision-OPD-baseline-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374
  log "FC4: launching answer-hint x unfiltered 150-step keep-all (GPU0-3)"
  MODEL_SIZE=2B CUDA_VISIBLE_DEVICES=0,1,2,3 TRAINER_N_GPUS_PER_NODE=4 \
    EXPERIMENT_NAME=Vision-OPD-baseline-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374 \
    ANSWER_VAL_TRAIN_FILE=$V/data/virl39k_train_noimg_unfiltered_1img.parquet \
    TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
    bash scripts/run_experiment_baseline.sh \
    trainer.max_actor_ckpt_to_keep=20 \
    data.filter_overlong_prompts=True trainer.total_training_steps=150 \
    > logs/fc4_150step_$(date +%Y%m%d_%H%M%S).log 2>&1
  if [ "$(step "$FC4")" -ge 150 ]; then
    for st in 30 60 90 120 150; do
      d="$FC4/global_step_${st}"
      [ -d "$d/actor" ] && [ ! -f "$d/config.json" ] && bash scripts/merge_checkpoint.sh "$d" >> "$LOG" 2>&1 && log "FC4 step${st} merged"
    done
    log "FC4 DONE + merged"
  else
    log "ERROR: FC4 at step $(step "$FC4")"
  fi
) &
PFC4=$!

# ---- L2 a-d (GPU4-7) ----
( eval8192 4 28870 "$Q2B" "l2a_vanilla_2b_eval8192" ) & PA=$!
( eval8192 5 28871 "$V/checkpoints/Vision-OPD-contrast-standard-uniformweight-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301829143/global_step_90" "l2b_ours_uniform_2b_eval8192" ) & PB=$!
( eval8192 6 28872 "$V/checkpoints/Vision-OPD-baseline-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301761390/global_step_90" "l2c_opsd_unfiltered_2b_eval8192" ) & PC=$!
( eval8192 7 28873 "$V/checkpoints/Vision-OPD-contrast-standard-uniformweight-Qwen3-VL-4B-virl39k-UNFILTERED1img-90step-trial301829143/global_step_90" "l2d_ours_uniform_4b_eval8192" ) & PD=$!
wait $PA $PB
# ---- L2e 两臂（GPU4/5, qwen35 shim）----
( eval8192 4 28874 "$Q35_4B" "l2e_vanilla_qwen35_eval8192" shim ) & PE1=$!
( eval8192 5 28875 "$V/checkpoints/Vision-OPD-contrast-standard-Qwen3.5-4B-virl39k-filtered-90step-trial301761390/global_step_90" "l2e_ours_qwen35_eval8192" shim ) & PE2=$!
wait $PC $PD $PE1 $PE2 $PFC4
log "=== FC4+L2 driver finished ==="
