#!/bin/bash
# 续跑：FCE 120组合批次，从被"不要跑judge"指令打断的地方(99/152完成)恢复，跳过已完成的100个
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
LOG=$V/logs/rejudge_batch_lowconc_resume.log
> "$LOG"
log() { echo "[$(date)] $*" | tee -a "$LOG"; }

declare -A DONE
DONE["fc1_uniform_unfiltered_step10/BLINK"]=1
DONE["fc1_uniform_unfiltered_step10/HRBench4K"]=1
DONE["fc1_uniform_unfiltered_step10/HRBench8K"]=1
DONE["fc1_uniform_unfiltered_step10/MMBench_DEV_EN"]=1
DONE["fc1_uniform_unfiltered_step10/MMStar"]=1
DONE["fc1_uniform_unfiltered_step10/VStarBench"]=1
DONE["fc1_uniform_unfiltered_step20/BLINK"]=1
DONE["fc1_uniform_unfiltered_step20/HRBench4K"]=1
DONE["fc1_uniform_unfiltered_step20/HRBench8K"]=1
DONE["fc1_uniform_unfiltered_step20/MMBench_DEV_EN"]=1
DONE["fc1_uniform_unfiltered_step20/MMStar"]=1
DONE["fc1_uniform_unfiltered_step20/VStarBench"]=1
DONE["fc1_uniform_unfiltered_step40/BLINK"]=1
DONE["fc1_uniform_unfiltered_step40/HRBench4K"]=1
DONE["fc1_uniform_unfiltered_step40/HRBench8K"]=1
DONE["fc1_uniform_unfiltered_step40/MMBench_DEV_EN"]=1
DONE["fc1_uniform_unfiltered_step40/MMStar"]=1
DONE["fc1_uniform_unfiltered_step40/VStarBench"]=1
DONE["fc1_uniform_unfiltered_step50/BLINK"]=1
DONE["fc1_uniform_unfiltered_step50/HRBench4K"]=1
DONE["fc1_uniform_unfiltered_step50/HRBench8K"]=1
DONE["fc1_uniform_unfiltered_step50/MMBench_DEV_EN"]=1
DONE["fc1_uniform_unfiltered_step50/MMStar"]=1
DONE["fc1_uniform_unfiltered_step50/VStarBench"]=1
DONE["fc1_uniform_unfiltered_step70/BLINK"]=1
DONE["fc1_uniform_unfiltered_step70/HRBench4K"]=1
DONE["fc1_uniform_unfiltered_step70/HRBench8K"]=1
DONE["fc1_uniform_unfiltered_step70/MMBench_DEV_EN"]=1
DONE["fc1_uniform_unfiltered_step70/MMStar"]=1
DONE["fc1_uniform_unfiltered_step70/VStarBench"]=1
DONE["fc1_uniform_unfiltered_step80/BLINK"]=1
DONE["fc1_uniform_unfiltered_step80/HRBench4K"]=1
DONE["fc1_uniform_unfiltered_step80/HRBench8K"]=1
DONE["fc1_uniform_unfiltered_step80/MMBench_DEV_EN"]=1
DONE["fc1_uniform_unfiltered_step80/MMStar"]=1
DONE["fc1_uniform_unfiltered_step80/VStarBench"]=1
DONE["fc4_opsd_unfiltered_step10/BLINK"]=1
DONE["fc4_opsd_unfiltered_step10/HRBench4K"]=1
DONE["fc4_opsd_unfiltered_step10/HRBench8K"]=1
DONE["fc4_opsd_unfiltered_step10/MMBench_DEV_EN"]=1
DONE["fc4_opsd_unfiltered_step10/MMStar"]=1
DONE["fc4_opsd_unfiltered_step10/VStarBench"]=1
DONE["fc4_opsd_unfiltered_step20/BLINK"]=1
DONE["fc4_opsd_unfiltered_step20/HRBench4K"]=1
DONE["fc4_opsd_unfiltered_step20/HRBench8K"]=1
DONE["fc4_opsd_unfiltered_step20/MMBench_DEV_EN"]=1
DONE["fc4_opsd_unfiltered_step20/MMStar"]=1
DONE["fc4_opsd_unfiltered_step20/VStarBench"]=1
DONE["fc4_opsd_unfiltered_step40/BLINK"]=1
DONE["fc4_opsd_unfiltered_step40/HRBench4K"]=1
DONE["fc4_opsd_unfiltered_step40/HRBench8K"]=1
DONE["fc4_opsd_unfiltered_step40/MMBench_DEV_EN"]=1
DONE["fc4_opsd_unfiltered_step40/MMStar"]=1
DONE["fc4_opsd_unfiltered_step40/VStarBench"]=1
DONE["fc4_opsd_unfiltered_step50/BLINK"]=1
DONE["fc4_opsd_unfiltered_step50/HRBench4K"]=1
DONE["fc4_opsd_unfiltered_step50/HRBench8K"]=1
DONE["fc4_opsd_unfiltered_step50/MMBench_DEV_EN"]=1
DONE["fc4_opsd_unfiltered_step50/MMStar"]=1
DONE["fc4_opsd_unfiltered_step50/VStarBench"]=1
DONE["fc4_opsd_unfiltered_step70/BLINK"]=1
DONE["fc4_opsd_unfiltered_step70/HRBench4K"]=1
DONE["fc4_opsd_unfiltered_step70/HRBench8K"]=1
DONE["fc4_opsd_unfiltered_step70/MMBench_DEV_EN"]=1
DONE["fc4_opsd_unfiltered_step70/MMStar"]=1
DONE["fc4_opsd_unfiltered_step70/VStarBench"]=1
DONE["fc4_opsd_unfiltered_step80/BLINK"]=1
DONE["fc4_opsd_unfiltered_step80/VStarBench"]=1
DONE["n4_uniformweight_qwen35_9b_unfiltered_step90/BLINK"]=1
DONE["n4_uniformweight_qwen35_9b_unfiltered_step90/HallusionBench"]=1
DONE["n4_uniformweight_qwen35_9b_unfiltered_step90/HRBench4K"]=1
DONE["n4_uniformweight_qwen35_9b_unfiltered_step90/HRBench8K"]=1
DONE["n4_uniformweight_qwen35_9b_unfiltered_step90/MMBench_DEV_EN"]=1
DONE["n4_uniformweight_qwen35_9b_unfiltered_step90/MMStar"]=1
DONE["n4_uniformweight_qwen35_9b_unfiltered_step90/POPE"]=1
DONE["n4_uniformweight_qwen35_9b_unfiltered_step90/VStarBench"]=1
DONE["ql1_uniform_qwen35_4b_unfiltered_len4096_step90/BLINK"]=1
DONE["ql1_uniform_qwen35_4b_unfiltered_len4096_step90/HallusionBench"]=1
DONE["ql1_uniform_qwen35_4b_unfiltered_len4096_step90/HRBench4K"]=1
DONE["ql1_uniform_qwen35_4b_unfiltered_len4096_step90/HRBench8K"]=1
DONE["ql1_uniform_qwen35_4b_unfiltered_len4096_step90/MMBench_DEV_EN"]=1
DONE["ql1_uniform_qwen35_4b_unfiltered_len4096_step90/MMStar"]=1
DONE["ql1_uniform_qwen35_4b_unfiltered_len4096_step90/POPE"]=1
DONE["ql1_uniform_qwen35_4b_unfiltered_len4096_step90/VStarBench"]=1
DONE["qs1_seed1234_qwen35_len4096_step90/BLINK"]=1
DONE["qs1_seed1234_qwen35_len4096_step90/HallusionBench"]=1
DONE["qs1_seed1234_qwen35_len4096_step90/HRBench4K"]=1
DONE["qs1_seed1234_qwen35_len4096_step90/HRBench8K"]=1
DONE["qs1_seed1234_qwen35_len4096_step90/MMBench_DEV_EN"]=1
DONE["qs1_seed1234_qwen35_len4096_step90/MMStar"]=1
DONE["qs1_seed1234_qwen35_len4096_step90/POPE"]=1
DONE["qs1_seed1234_qwen35_len4096_step90/VStarBench"]=1
DONE["s2c_answerhint_seed777_unfiltered_step90/BLINK"]=1
DONE["s2c_answerhint_seed777_unfiltered_step90/HallusionBench"]=1
DONE["s2c_answerhint_seed777_unfiltered_step90/HRBench4K"]=1
DONE["s2c_answerhint_seed777_unfiltered_step90/HRBench8K"]=1
DONE["s2c_answerhint_seed777_unfiltered_step90/MMBench_DEV_EN"]=1
DONE["s2c_answerhint_seed777_unfiltered_step90/MMStar"]=1
DONE["s2c_answerhint_seed777_unfiltered_step90/POPE"]=1
DONE["s2c_answerhint_seed777_unfiltered_step90/VStarBench"]=1

declare -a JOBS=()
for st in 10 20 40 50 70 80 100 110 130 140; do
  for fam in fc1 fc4; do
    if [ "$fam" = "fc1" ]; then mn="fc1_uniform_unfiltered_step${st}"; else mn="fc4_opsd_unfiltered_step${st}"; fi
    for ds in BLINK MMStar MMBench_DEV_EN VStarBench HRBench4K HRBench8K; do
      key="$mn/$ds"
      [ -n "${DONE[$key]:-}" ] && continue
      JOBS+=("$mn:$ds")
    done
  done
done
log "remaining jobs: ${#JOBS[@]}"

n=${#JOBS[@]}
idx=0
while [ $idx -lt $n ]; do
  batch=()
  for i in 1 2 3; do
    [ $idx -ge $n ] && break
    batch+=("${JOBS[$idx]}")
    idx=$((idx+1))
  done
  pids=()
  for item in "${batch[@]}"; do
    IFS=: read -r mn ds <<< "$item"
    ( bash "$V/scripts/rejudge_one.sh" "$mn" "$ds" 6 >> "$LOG" 2>&1 ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  log "batch done ($idx/$n)"
done
log "=== rejudge_batch_lowconc_resume all done ==="
