#!/bin/bash
# usage: rejudge_one.sh <model_name> <dataset> [nproc]
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
VLM=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
SHIM=$V/.syspkg_shim
KEYCONF=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/config/key.conf
MN="$1"; DS="$2"; NPROC="${3:-8}"
D=$(ls -d "$VLM"/outputs_vllm_curated/${MN}*_qwen3vl2b_temp0_4096_generic_eval/normal_scoring 2>/dev/null | head -1)
[ -z "$D" ] && { echo "NO DIR for $MN"; exit 1; }
pred=$(ls "$D"/${MN}*_${DS}_normal.xlsx 2>/dev/null | grep -v gpt | head -1)
[ -z "$pred" ] && { echo "NO PRED for $MN/$DS"; exit 1; }
# clear stale judge caches (both cache shapes used across dataset types)
rm -f "$D"/${MN}*_${DS}_normal_gpt-*_result.pkl "$D"/${MN}*_${DS}_normal_gpt-*.pkl \
      "$D"/${MN}*_${DS}_normal_acc.csv "$D"/${MN}*_${DS}_normal_gpt-*_score.csv \
      "$D"/${MN}*_${DS}_normal_auxmatch.xlsx "$D"/${MN}*_${DS}_normal_score.csv \
      "$D"/${MN}*_${DS}_normal_gpt-*.xlsx
cd "$VLM"
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 all_proxy=socks5h://127.0.0.1:1080
export LMUData=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/LMUData
logf="$D/$(basename "$pred" .xlsx)_gpt-5.4-mini-2026-03-17.log"
PYTHONPATH="$SHIM:$VLM" /usr/bin/python -u tools/run_normal_eval.py \
  --dataset "$DS" --prediction-file "$pred" \
  --judge gpt-5.4-mini-2026-03-17 --provider tiktok_azure \
  --nproc "$NPROC" --retry 12 --timeout 900 --temperature 0.0 \
  --key-conf "$KEYCONF" > "$logf" 2>&1
rc=$?
if grep -q "not working properly" "$logf"; then
  echo "$MN/$DS: STILL FELL BACK TO EXACT-MATCH (rc=$rc)"
  exit 2
elif [ $rc -eq 0 ] && grep -q RESULT_JSON "$logf"; then
  echo "$MN/$DS: OK -- $(grep RESULT_JSON "$logf" | head -c 200)"
else
  echo "$MN/$DS: FAILED rc=$rc, see $logf"
  exit 1
fi
