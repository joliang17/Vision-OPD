#!/usr/bin/env bash
# Re-score JSD step90 predictions (no GPU, no re-inference needed).
# Root cause of the original garbage scores: Azure judge hit sustained 429
# QPM limits (too many machines/mlx jobs sharing the endpoint), VLMEvalKit
# silently downgraded MCQ scoring to exact matching, which fails long-CoT
# answers wholesale. Re-run the judge at low concurrency with generous
# retries so it survives rate-limit windows.
set -uo pipefail
cd "$(dirname "$0")/../../VLMEvalKit"
LOG="../Vision-OPD/logs/rescore_jsd_step90.log"
D="outputs_api_server/jsd_2b_virl39k_90step_step90_eval/normal_scoring"
MODEL_NAME=jsd_2b_virl39k_90step_step90
export KEY_CONF=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/config/key.conf
# judge needs real internet -> proxy must be SET on this box
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890
unset no_proxy NO_PROXY

log() { echo "[$(date)] $*" | tee -a "$LOG"; }

log "=== JSD re-score started ==="
for ds in BLINK MMStar MMBench_DEV_EN VStarBench MathVista_MINI HRBench4K HRBench8K POPE HallusionBench; do
  pred="${D}/${MODEL_NAME}_${ds}_normal.xlsx"
  if [ ! -f "${pred}" ]; then
    log "SKIP ${ds}: no prediction file"
    continue
  fi
  # wipe stale judge caches so the rescore doesn't silently reuse the
  # exact-match-era results (HallusionBench/POPE keep _auxmatch/_score
  # files, MathVista keeps .pkl -- the known cache pitfall from 2026-07-14)
  rm -f "${D}/${MODEL_NAME}_${ds}_normal_"*.pkl \
        "${D}/${MODEL_NAME}_${ds}_normal_auxmatch.xlsx" \
        "${D}/${MODEL_NAME}_${ds}_normal_score.csv" \
        "${D}/${MODEL_NAME}_${ds}_normal_acc.csv" \
        "${D}/${MODEL_NAME}_${ds}_normal_gpt-5.4-mini"*.xlsx 2>/dev/null
  log "re-scoring ${ds} (nproc=2, retry=12)"
  /usr/bin/python -u tools/run_normal_eval.py \
    --dataset "${ds}" \
    --prediction-file "${pred}" \
    --judge gpt-5.4-mini-2026-03-17 \
    --provider tiktok_azure \
    --nproc 2 \
    --retry 12 \
    --timeout 600 \
    --temperature 0.0 \
    --key-conf "${KEY_CONF}" \
    >> "$LOG" 2>&1 || log "WARNING: ${ds} rescore failed, check log"
  grep -a "RESULT_JSON=" "$LOG" | tail -1
done
log "=== JSD re-score done ==="
