#!/bin/bash
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
OPSD=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "$OPSD"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
CKPT="$V/checkpoints/Vision-OPD-grpo-from-ours-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-1ep-trial301783374/global_step_194"
NAME=g1_grpo_from_ours_uniform_step194
PORT=28901
LOG="$V/logs/g1_zoombench.log"
log(){ echo "[$(date)] $*" | tee -a "$LOG"; }

log "serving on GPU1 port $PORT"
CUDA_VISIBLE_DEVICES=1 vllm serve "$CKPT" \
  --served-model-name "$NAME" --host 0.0.0.0 --port $PORT \
  --trust-remote-code --enforce-eager --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --dtype bfloat16 --limit-mm-per-prompt '{"image":8}' \
  > "$V/logs/g1_zoom_serve.log" 2>&1 &
SERVEPID=$!
for i in $(seq 1 60); do
  curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1 && break
  sleep 10
done
log "server up, starting infer"
cd "$V/eval"
python3 infer.py --benchmark zoombench --benchmark_json "$PWD/zoombench.json" \
  --out_dir model_answer --model_name "$NAME" --seed 42 \
  --api_base "http://127.0.0.1:$PORT/v1/" --api_key EMPTY \
  --model_id "$NAME" --max_tokens 8192 --max_retries 3 --parallel_workers 256 \
  >> "$LOG" 2>&1
log "infer done, starting judge"
source "$OPSD/config/key.conf"
python3 judge_qwenlm.py --benchmark zoombench --model "$NAME" \
  --api_base "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl" \
  --api_key "${AZURE_GPT_API_KEY}" --api_type azure --api_version "2024-02-01" \
  --judge_model "gpt-5.4-mini-2026-03-17" --judge_max_tokens 2048 \
  >> "$LOG" 2>&1
log "judge done, computing acc"
python3 cal_acc.py --benchmark zoombench --judge_json "judge/zoombench/${NAME}_answer.jsonl" \
  --benchmark_json "$PWD/zoombench.json" | tee -a "$LOG"
kill $SERVEPID 2>/dev/null
log "=== G1 zoombench done ==="
