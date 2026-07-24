#!/bin/bash
# 低并发(3路)批量重判：Stage1剩余4点(N4/S2c/QL1/QS1)全8数据集 + FCE 20个新点全6数据集
# 每次只3路同时，避开working()探测被并发限流的坑（教训见queue.md 07-21条目）
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
LOG=$V/logs/rejudge_batch_lowconc.log
> "$LOG"
log() { echo "[$(date)] $*" | tee -a "$LOG"; }

declare -a JOBS=()
for mn in n4_uniformweight_qwen35_9b_unfiltered_step90 s2c_answerhint_seed777_unfiltered_step90 \
          ql1_uniform_qwen35_4b_unfiltered_len4096_step90 qs1_seed1234_qwen35_len4096_step90; do
  for ds in BLINK MMStar MMBench_DEV_EN VStarBench HRBench4K HRBench8K POPE HallusionBench; do
    JOBS+=("$mn:$ds")
  done
done
for st in 10 20 40 50 70 80 100 110 130 140; do
  for fam in fc1 fc4; do
    if [ "$fam" = "fc1" ]; then mn="fc1_uniform_unfiltered_step${st}"; else mn="fc4_opsd_unfiltered_step${st}"; fi
    for ds in BLINK MMStar MMBench_DEV_EN VStarBench HRBench4K HRBench8K; do
      JOBS+=("$mn:$ds")
    done
  done
done
log "total jobs: ${#JOBS[@]}"

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
log "=== rejudge_batch_lowconc all done ==="
