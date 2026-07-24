#!/bin/bash
# FC1 (ours) 6 个被 exact-match 污染的 Hallu 点重判 (step 10/20/40/50/70/80)
# 纯 judge/API,不占 GPU,串行防 429。07-23 devbox。
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
LOG=logs/fce_fc1_hallu_rejudge_$(date +%Y%m%d_%H%M%S).log
for s in 10 20 40 50 70 80; do
  MN="fc1_uniform_unfiltered_step${s}_server"
  echo "===== rejudge $MN HallusionBench =====" | tee -a "$LOG"
  bash scripts/rejudge_one.sh "$MN" HallusionBench 8 2>&1 | tee -a "$LOG"
done
echo "=== FC1 Hallu 6点重判完成 ===" | tee -a "$LOG"
