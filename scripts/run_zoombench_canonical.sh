#!/usr/bin/env bash
# Canonical ZoomBench eval per Vision-OPD/CLAUDE.md "ZoomBench — canonical eval (default)":
# native pipeline (infer.py -> judge_qwenlm.py -> cal_acc.py) with a REAL external GPT judge
# (gpt-5.4-mini via Azure), not self-judge. Do not use eval/run_zoombench.sh for anything that
# needs to go in compare_vaopd_0701.md -- that script self-judges and is not the canonical path
# (confirmed to produce a ~30pp inflated false-positive rate on at least one checkpoint,
# 2026-07-11/12 postmortem).
#
# Usage: bash scripts/run_zoombench_canonical.sh <model_path> <model_name> <gpu> <port>
set -euo pipefail

MODEL_PATH="${1:?usage: run_zoombench_canonical.sh <model_path> <model_name> <gpu> <port>}"
MODEL_NAME="${2:?usage: run_zoombench_canonical.sh <model_path> <model_name> <gpu> <port>}"
GPU="${3:-0}"
PORT="${4:-8000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPSD_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
KEY_CONF="${OPSD_ROOT}/config/key.conf"
AZURE_GPT_API_KEY="$(grep -m1 '^AZURE_GPT_API_KEY=' "${KEY_CONF}" | cut -d= -f2- | tr -d '"')"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="localhost,127.0.0.1,::1"; export NO_PROXY="${no_proxy}"

echo "[serve] GPU=${GPU} port=${PORT} model=${MODEL_PATH}"
CUDA_VISIBLE_DEVICES="${GPU}" nohup vllm serve "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" --host 0.0.0.0 --port "${PORT}" \
  --trust-remote-code --enforce-eager --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --dtype bfloat16 --limit-mm-per-prompt '{"image":8}' \
  > "/tmp/vllm_canonical_${MODEL_NAME}_${PORT}.log" 2>&1 &
SERVE_PID=$!
trap 'kill ${SERVE_PID} 2>/dev/null || true' EXIT

echo "[serve] waiting for readiness..."
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { echo "[serve] ready after ${i}0s"; break; }
  kill -0 "${SERVE_PID}" 2>/dev/null || { echo "[serve] died"; tail -40 "/tmp/vllm_canonical_${MODEL_NAME}_${PORT}.log"; exit 1; }
  sleep 10
done

cd "${REPO_ROOT}/eval"

echo "[1/3] infer..."
python3 infer.py --benchmark zoombench --benchmark_json "$PWD/zoombench.json" \
  --out_dir model_answer --model_name "${MODEL_NAME}" --seed 42 \
  --api_base "http://127.0.0.1:${PORT}/v1/" --api_key EMPTY \
  --model_id "${MODEL_NAME}" --max_tokens 8192 --max_retries 3 --parallel_workers 256

echo "[2/3] judge (canonical: gpt-5.4-mini via azure, judge_max_tokens=2048)..."
python3 judge_qwenlm.py --benchmark zoombench --model "${MODEL_NAME}" \
  --api_base "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl" \
  --api_key "${AZURE_GPT_API_KEY}" --api_type azure --api_version "2024-02-01" \
  --judge_model "gpt-5.4-mini-2026-03-17" --judge_max_tokens 2048

echo "[3/3] accuracy..."
python3 cal_acc.py --benchmark zoombench \
  --judge_json "judge/zoombench/${MODEL_NAME}_answer.jsonl" \
  --benchmark_json "$PWD/zoombench.json"
