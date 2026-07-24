#!/usr/bin/env bash
# Usage: run_model_eval.sh <MODEL_TAG> <OPENAI_MODEL_ID>
# Resume-safe: skips benchmarks already scored; infer.py resumes partial runs.
set -uo pipefail
cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/eval

MODEL_TAG="$1"
OPENAI_MODEL_ID="$2"
API_BASE="${API_BASE:-http://localhost:8000/v1/}"

export JUDGE_AZURE_API_VERSION="2024-02-01"
JUDGE_BASE="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl"
JUDGE_KEY="er2VRenl0rLBHoGb3epo4sxVEsFQSzNt"
JUDGE_MODEL="gpt-5.4-2026-03-05"

declare -A M=(
  [vstar]=vstar.json
  [hrbench-4k]=hr_bench_4k.json
  [hrbench-8k]=hr_bench_8k.json
  [zoombench]=zoombench.json
  [mme-realworld]=MME_RealWorld.json
  [mme-realworld-cn]=MME_RealWorld_CN.json
)
ORDER=(vstar hrbench-4k hrbench-8k zoombench mme-realworld-cn)

RESULT="logs/results_${MODEL_TAG}.txt"
touch "$RESULT"

for b in "${ORDER[@]}"; do
  json="${M[$b]}"
  # Resume: skip benchmarks already scored in the results file
  if grep -q "^=== ${b} ===$" "$RESULT"; then
    echo "######## [$MODEL_TAG] SKIP $b (already scored) ########"
    continue
  fi
  waited=0
  while [ ! -f "$json" ]; do
    echo "[$b] waiting for data ($json)... ${waited}s"
    sleep 15; waited=$((waited+15))
  done

  echo "######## [$MODEL_TAG] INFER $b $(date) ########"
  python3 infer.py --benchmark "$b" --benchmark_json "$PWD/$json" --out_dir model_answer \
    --model_name "$MODEL_TAG" --seed 42 --api_base "$API_BASE" --api_key EMPTY \
    --model_id "$OPENAI_MODEL_ID" --max_tokens 32768 --max_retries 3 --parallel_workers 256

  echo "######## [$MODEL_TAG] JUDGE $b $(date) ########"
  python3 judge_qwenlm.py --benchmark "$b" --model "$MODEL_TAG" \
    --api_base "$JUDGE_BASE" --api_key "$JUDGE_KEY" --api_type azure \
    --api_version "$JUDGE_AZURE_API_VERSION" --judge_model "$JUDGE_MODEL" \
    --judge_max_tokens 2048

  echo "######## [$MODEL_TAG] ACC $b $(date) ########"
  acc=$(python3 cal_acc.py --benchmark "$b" \
        --judge_json "judge/$b/${MODEL_TAG}_answer.jsonl" \
        --benchmark_json "$PWD/$json")
  echo "$acc"
  { echo "=== $b ==="; echo "$acc"; } >> "$RESULT"
done
echo "MODEL_EVAL_DONE $MODEL_TAG"
