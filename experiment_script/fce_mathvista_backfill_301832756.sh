#!/bin/bash
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
VLM=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
SHIM=$V/.syspkg_shim
KEYCONF=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/config/key.conf
LOG=$V/logs/fce_mathvista_backfill_driver.log
log() { echo "[$(date)] $*" | tee -a "$LOG"; }
cd "$VLM"
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 all_proxy=socks5h://127.0.0.1:1080
for st in 10 20 40 50 70 80 100 110 130 140; do
  for fam in fc1 fc4; do
    if [ "$fam" = "fc1" ]; then mn="fc1_uniform_unfiltered_step${st}"; else mn="fc4_opsd_unfiltered_step${st}"; fi
    D="outputs_vllm_curated/${mn}_server_qwen3vl2b_temp0_4096_generic_eval"
    pred="$D/normal_scoring/${mn}_MathVista_MINI_normal.xlsx"
    if [ ! -f "$pred" ]; then log "$mn: no xlsx pred found, skip"; continue; fi
    logf="$D/normal_scoring/${mn}_MathVista_MINI_normal_gpt-5.4-mini-2026-03-17.log"
    PYTHONPATH="$SHIM:$VLM" /usr/bin/python -u tools/run_normal_eval.py \
      --dataset MathVista_MINI --prediction-file "$pred" \
      --judge gpt-5.4-mini-2026-03-17 --provider tiktok_azure \
      --nproc 8 --retry 12 --timeout 900 --temperature 0.0 \
      --key-conf "$KEYCONF" > "$logf" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && grep -q RESULT_JSON "$logf"; then
      log "$mn: OK -- $(grep RESULT_JSON "$logf" | head -c 150)"
    else
      log "$mn: FAILED rc=$rc, see $logf"
    fi
  done
done
log "=== fce mathvista backfill done ==="
