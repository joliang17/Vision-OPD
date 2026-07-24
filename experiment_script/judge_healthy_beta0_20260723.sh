#!/bin/bash
# 直接judge β=0 各checkpoint的健康预测(原始栈,不占GPU),尽快出分看问题
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
VLM=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
KEYCONF=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/config/key.conf
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
unset ALL_PROXY all_proxy
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
export no_proxy="localhost,127.0.0.1,::1,byteintl.net,aidp-i18ntt-sg.byteintl.net"; export NO_PROXY="$no_proxy"
export LMUData=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/LMUData
export PYTHONPATH=$V/.syspkg_shim:$VLM
cd "$VLM"
judge_one(){
  local st=$1 ds=$2
  local root=outputs_vllm_curated/beta0_matchfc1_step${st}_qwen3vl2b_temp0_4096_generic_eval
  local pred=$(find "$root" -name "*_${ds}.xlsx" 2>/dev/null|grep -v gpt|head -1)
  [ -z "$pred" ] && { echo "step$st $ds: 无预测"; return; }
  mkdir -p "$root/normal_scoring"
  local dst="$root/normal_scoring/beta0_matchfc1_step${st}_${ds}_normal.xlsx"
  cp -f "$pred" "$dst"
  local lg="$V/logs/jh_beta0_step${st}_${ds}.log"
  /usr/bin/python -u tools/run_normal_eval.py --dataset "$ds" --prediction-file "$dst" \
    --judge gpt-5.4-mini-2026-03-17 --provider tiktok_azure --nproc 16 --retry 12 --timeout 900 \
    --temperature 0.0 --key-conf "$KEYCONF" > "$lg" 2>&1
  grep -q RESULT_JSON "$lg" && echo "step$st $ds: ✅" || echo "step$st $ds: ⚠️($(tail -1 "$lg"|cut -c1-40))"
}
export -f judge_one; export V VLM KEYCONF
# 健康benchmark清单(并行,但限总数防429)
tasks="30:BLINK 30:MMStar 30:VStarBench 30:MathVista_MINI 30:HRBench4K \
60:BLINK 60:MMStar 60:VStarBench 60:MathVista_MINI 60:HRBench4K \
90:BLINK 90:MMStar 90:VStarBench 90:MathVista_MINI \
120:BLINK 120:MMStar 120:VStarBench 120:MathVista_MINI \
150:BLINK 150:MMStar 150:VStarBench"
echo "$tasks" | tr ' ' '\n' | grep -v '^$' | xargs -P 6 -I{} bash -c 'judge_one ${0%:*} ${0#*:}' {}
echo "=== 健康预测judge完成 ==="
