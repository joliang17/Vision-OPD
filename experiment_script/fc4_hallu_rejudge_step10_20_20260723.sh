#!/bin/bash
# FC4(baseline) step10/20 Hallu 干净重判(正确proxy)—— 确认 step20=55.62 是否真实
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
VLM=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
KEYCONF=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/config/key.conf
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
unset ALL_PROXY all_proxy
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
export no_proxy="localhost,127.0.0.1,::1,byteintl.net,aidp-i18ntt-sg.byteintl.net"
export NO_PROXY="$no_proxy"
export LMUData=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/LMUData
export PYTHONPATH=$V/.syspkg_shim_nonumpy:$VLM
cd "$VLM"
TS=$(date +%Y%m%d_%H%M%S)
rejudge(){
  local s=$1
  local D="outputs_vllm_curated/fc4_opsd_unfiltered_step${s}_qwen3vl2b_temp0_4096_generic_eval/normal_scoring"
  [ -d "$D" ] || D="outputs_vllm_curated/fc4_opsd_unfiltered_step${s}_server_qwen3vl2b_temp0_4096_generic_eval/normal_scoring"
  local pred=$(ls "$D"/fc4_opsd_unfiltered_step${s}*_HallusionBench_normal.xlsx 2>/dev/null|grep -v gpt|head -1)
  [ -z "$pred" ] && { echo "step$s NO PRED ($D)"; return; }
  rm -f "$D"/*HallusionBench*_auxmatch.xlsx "$D"/*HallusionBench*_tmp.pkl \
        "$D"/*HallusionBench*_acc.csv "$D"/*HallusionBench*_score.csv \
        "$D"/*HallusionBench*_gpt-*.pkl "$D"/*HallusionBench*_gpt-*.xlsx "$D"/*HallusionBench*_gpt-*.csv
  local lg="$V/logs/rejudge_fc4_hallu_step${s}_${TS}.log"
  /usr/bin/python -u tools/run_normal_eval.py --dataset HallusionBench --prediction-file "$pred" \
    --judge gpt-5.4-mini-2026-03-17 --provider tiktok_azure --nproc 32 --retry 12 --timeout 900 \
    --temperature 0.0 --key-conf "$KEYCONF" > "$lg" 2>&1
  if grep -q "not working" "$lg"; then echo "step$s ⚠️回退"; 
  elif grep -q RESULT_JSON "$lg"; then echo "step$s ✅ $(grep RESULT_JSON "$lg"|tail -1|grep -oE '"aAcc": [0-9.]+, "fAcc": [0-9.]+, "qAcc": [0-9.]+'|head -1)";
  else echo "step$s FAILED"; fi
}
export -f rejudge; export V VLM KEYCONF TS
for s in 10 20; do rejudge "$s" & done
wait
echo "=== FC4 step10/20 重判完成 ==="
