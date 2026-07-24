#!/bin/bash
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
VLM=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
SHIM=$V/.syspkg_shim
KEYCONF=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/config/key.conf
LOG=$V/logs/mathvista_backfill_driver.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
cd "$VLM"
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 all_proxy=socks5h://127.0.0.1:1080
for f in alpha0_matched_baseline_step90 n4_uniformweight_qwen35_9b_unfiltered_step90 s2c_answerhint_seed777_unfiltered_step90 ql1_uniform_qwen35_4b_unfiltered_len4096_step90 qs1_seed1234_qwen35_len4096_step90; do
  D="outputs_vllm_curated/${f}_server_qwen3vl2b_temp0_4096_generic_eval"
  pred=$(find "$D/${f}" -name "${f}_MathVista_MINI.xlsx" 2>/dev/null | sort | tail -1)
  if [ -z "$pred" ]; then log "$f: NO PREDICTION FOUND, skip"; continue; fi
  outpred="$D/normal_scoring/${f}_MathVista_MINI_normal.xlsx"
  cp -f "$pred" "$outpred"
  logf="$D/normal_scoring/${f}_MathVista_MINI_normal_gpt-5.4-mini-2026-03-17.log"
  PYTHONPATH="$SHIM:$VLM" /usr/bin/python -u tools/run_normal_eval.py \
    --dataset MathVista_MINI --prediction-file "$outpred" \
    --judge gpt-5.4-mini-2026-03-17 --provider tiktok_azure \
    --nproc 8 --retry 12 --timeout 900 --temperature 0.0 \
    --key-conf "$KEYCONF" > "$logf" 2>&1
  log "$f: rc=$? -- $(grep RESULT_JSON "$logf" | head -c 300)"
done
log "=== mathvista backfill done ==="
