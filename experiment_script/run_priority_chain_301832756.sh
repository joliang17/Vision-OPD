#!/bin/bash
# 2026-07-21: N4/S2c/QL1 eval 收尾后按用户排定优先级接:
#   1. α=0 matched baseline eval (9-bench, 最高价值)
#   2. FCE 细曲线: 先补merge FC1/FC4 中间10档, 再 30点×7-bench
#   3. QS1 eval (9-bench, seed对照, 已降优先)
# 全程走 conda qwen35 (本机系统栈三连坏的教训, 见 opsd/CLAUDE.md——即使是 plain Qwen3-VL 也用conda)
set -uo pipefail
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=$OPSD/Vision-OPD
cd "$V"
LOG=logs/priority_chain_driver_301832756.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
EVAL_VER=server_qwen3vl2b_temp0_4096_generic
DATASETS9=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench
DATASETS7=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K

source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35

run_lane() { # $1=gpu $2=port $3=model_path $4=model_name $5=datasets
  local gpu=$1 port=$2 mp=$3 mn=$4 ds=$5
  (
    cd "$OPSD/VLMEvalKit"
    env BACKEND=vllm_server PORT=$port EVAL_SETTING_VERSION=$EVAL_VER \
      JUDGE_API_NPROC=2 JUDGE_RETRY=12 REQUEST_TIMEOUT=900 RETRY=4 \
      MODEL_PATH=$mp MODEL_NAME=$mn DATASETS=$ds GPU_IDS=$gpu \
      bash shell_scripts/eval_model_temp0_4096.sh \
      > $V/logs/eval_local_${mn}.log 2>&1
    echo "[$(date)] lane $mn (gpu$gpu) exited rc=$?" >> "$LOG"
  ) &
}

log "=== waiting for N4/S2c/QL1 batch to fully clear ==="
while ps aux | grep -v grep | grep -q run_eval_n4_s2c_ql1_301832756.sh; do sleep 60; done
log "N4/S2c/QL1 batch cleared"

CKPT=$V/checkpoints

# ---- 1. α=0 matched baseline (9-bench) ----
log "=== stage 1: alpha0 matched baseline eval ==="
run_lane 0 28930 "$CKPT/Vision-OPD-contrast-alpha0-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-90step-trial301832790/global_step_90" alpha0_matched_baseline_step90 $DATASETS9
wait
log "stage 1 done"

# ---- 2a. 补 merge FC1/FC4 中间10档 ----
log "=== stage 2a: merge FC1/FC4 中间10档 ==="
FC1=$CKPT/Vision-OPD-contrast-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374
FC4=$CKPT/Vision-OPD-baseline-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374
for d in "$FC1" "$FC4"; do
  for st in 10 20 40 50 70 80 100 110 130 140; do
    dd="$d/global_step_${st}"
    [ -d "$dd/actor" ] && [ ! -f "$dd/config.json" ] && PYTHONNOUSERSITE=1 bash scripts/merge_checkpoint.sh "$dd" >> "$LOG" 2>&1 && log "merged $(basename $d)/global_step_${st}" || log "MERGE FAILED $(basename $d)/global_step_${st}"
  done
done
log "stage 2a merge done"

# ---- 2b. FCE 30点 × 7-bench, 8宽波次 ----
log "=== stage 2b: FCE 30点eval (8宽波次) ==="
STEPS="10 20 30 40 50 60 70 80 90 100 110 120 130 140 150"
i=0
declare -a QUEUE
for st in $STEPS; do QUEUE+=("fc1:$st"); done
for st in $STEPS; do QUEUE+=("fc4:$st"); done
n=${#QUEUE[@]}
idx=0
while [ $idx -lt $n ]; do
  batch=()
  for g in 0 1 2 3 4 5 6 7; do
    [ $idx -ge $n ] && break
    batch+=("${QUEUE[$idx]}:$g")
    idx=$((idx+1))
  done
  for item in "${batch[@]}"; do
    IFS=: read -r fam st gpu <<< "$item"
    if [ "$fam" = "fc1" ]; then d="$FC1"; mn="fc1_uniform_unfiltered_step${st}"; else d="$FC4"; mn="fc4_opsd_unfiltered_step${st}"; fi
    port=$((29000 + gpu))
    run_lane "$gpu" "$port" "$d/global_step_${st}" "$mn" "$DATASETS7"
  done
  wait
  log "FCE wave done (${#batch[@]} lanes)"
done
log "stage 2b FCE eval done (30点)"

# ---- 3. QS1 eval (9-bench) ----
log "=== stage 3: QS1 eval ==="
run_lane 0 28940 "$CKPT/Vision-OPD-contrast-uniform-seed1234-Qwen3.5-4B-virl39k-UNFILTERED1img-90step-len4096-trial301832756/global_step_90" qs1_seed1234_qwen35_len4096_step90 $DATASETS9
wait
log "stage 3 done"

log "=== priority chain finished ==="
