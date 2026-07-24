#!/bin/bash
# FC1 6个 Hallu 点重判 —— 修正 proxy(unset all_proxy + no_proxy含byteintl),judge已验证 working()=True
# 之前 rejudge_one.sh 硬设 all_proxy=socks5h 导致回退 exact-match(fAcc逐位相同的2.601垃圾值)
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
  local D="outputs_vllm_curated/fc1_uniform_unfiltered_step${s}_server_qwen3vl2b_temp0_4096_generic_eval/normal_scoring"
  local pred=$(ls "$D"/fc1_uniform_unfiltered_step${s}_server*_HallusionBench_normal.xlsx 2>/dev/null | grep -v gpt | head -1)
  [ -z "$pred" ] && { echo "step$s: NO PRED"; return; }
  # 清所有污染缓存
  rm -f "$D"/*HallusionBench*_auxmatch.xlsx "$D"/*HallusionBench*_tmp.pkl \
        "$D"/*HallusionBench*_acc.csv "$D"/*HallusionBench*_score.csv \
        "$D"/*HallusionBench*_gpt-*.pkl "$D"/*HallusionBench*_gpt-*.xlsx "$D"/*HallusionBench*_gpt-*.csv
  local lg="$V/logs/rejudge2_fc1_hallu_step${s}_${TS}.log"
  /usr/bin/python -u tools/run_normal_eval.py \
    --dataset HallusionBench --prediction-file "$pred" \
    --judge gpt-5.4-mini-2026-03-17 --provider tiktok_azure \
    --nproc 32 --retry 12 --timeout 900 --temperature 0.0 \
    --key-conf "$KEYCONF" > "$lg" 2>&1
  if grep -q "not working" "$lg"; then echo "step$s: ⚠️仍回退exact-match"
  elif grep -q RESULT_JSON "$lg"; then
    echo "step$s: ✅ $(grep RESULT_JSON "$lg"|tail -1|grep -oE 'aAcc[^,]*,[^,]*fAcc[^,]*,[^,]*qAcc[^}]*'|head -1)"
  else echo "step$s: FAILED (see $lg)"; fi
}
export -f rejudge; export V VLM KEYCONF TS
for s in 10 20 40 50 70 80; do rejudge "$s" & done
wait
echo "=== FC1 Hallu 6点(修正proxy)重判完成 ==="
