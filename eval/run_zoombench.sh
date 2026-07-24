#!/usr/bin/env bash
# Run ZoomBench evaluation for a given checkpoint.
#
# Usage:
#   bash eval/run_zoombench.sh <model_path> <model_name> [gpu] [port]
#
# Examples:
#   bash eval/run_zoombench.sh \
#     /path/to/checkpoints/global_step_65 \
#     VisionOPD-Qwen3-VL-2B-Instruct
#
#   bash eval/run_zoombench.sh /path/to/model MyModel 1 8001
#
# Environment overrides (optional):
#   GPU=0          CUDA device index (default: 0)
#   PORT=8000      vLLM server port (default: 8000)
#   MAX_TOKENS=128 inference max tokens (default: 128)
#   PARALLEL_WORKERS=64

set -euo pipefail

# ── Args ────────────────────────────────────────────────────────────────────
MODEL_PATH="${1:?Usage: $0 <model_path> <model_name> [gpu] [port]}"
MODEL_NAME="${2:?Usage: $0 <model_path> <model_name> [gpu] [port]}"
GPU="${3:-${GPU:-0}}"
PORT="${4:-${PORT:-8000}}"

MAX_TOKENS="${MAX_TOKENS:-128}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_JSON="$SCRIPT_DIR/zoombench_mcq.json"

# ── Proxy fix ────────────────────────────────────────────────────────────────
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"

# ── Derived names ────────────────────────────────────────────────────────────
# Use last two path components as a short tag to avoid "/" in filenames
MODEL_TAG="${MODEL_NAME//\//_}"
VLLM_LOG="/tmp/vllm_${MODEL_TAG}_${PORT}.log"

echo "=========================================="
echo "Model path : $MODEL_PATH"
echo "Model name : $MODEL_NAME"
echo "GPU        : $GPU   Port: $PORT"
echo "=========================================="

# ── Start vLLM ──────────────────────────────────────────────────────────────
> "$VLLM_LOG"
CUDA_VISIBLE_DEVICES="$GPU" nohup vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --port "$PORT" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  --trust-remote-code \
  >> "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Kill server on exit (normal or error)
trap 'echo "Stopping vLLM (PID $VLLM_PID)..."; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null; echo "Stopped."' EXIT

echo "vLLM PID: $VLLM_PID  (log: $VLLM_LOG)"
echo "Waiting for server..."
until grep -q "Application startup complete" "$VLLM_LOG" 2>/dev/null; do
  sleep 10
  kill -0 $VLLM_PID 2>/dev/null || { echo "ERROR: vLLM server died!"; tail -30 "$VLLM_LOG"; exit 1; }
done
echo "Server ready!"

# ── Inference ────────────────────────────────────────────────────────────────
echo
echo "[1/3] Running inference..."
python3 "$SCRIPT_DIR/infer.py" \
  --benchmark zoombench \
  --benchmark_json "$BENCH_JSON" \
  --out_dir "$SCRIPT_DIR/model_answer" \
  --model_name "${MODEL_TAG}_mcq_seed42" \
  --seed 42 \
  --api_base "http://localhost:${PORT}/v1/" \
  --api_key "EMPTY" \
  --model_id "$MODEL_NAME" \
  --max_tokens "$MAX_TOKENS" \
  --max_retries 3 \
  --parallel_workers "$PARALLEL_WORKERS"

# ── Judge ────────────────────────────────────────────────────────────────────
echo
echo "[2/3] Running judge (self-judge with same model)..."
python3 "$SCRIPT_DIR/judge_qwenlm.py" \
  --benchmark zoombench \
  --model "${MODEL_TAG}_mcq_seed42" \
  --answer_dir "$SCRIPT_DIR/model_answer" \
  --api_base "http://localhost:${PORT}/v1/" \
  --api_key "EMPTY" \
  --judge_model "$MODEL_NAME" \
  --judge_max_tokens 64

# ── Accuracy ─────────────────────────────────────────────────────────────────
echo
echo "[3/3] Accuracy:"
python3 "$SCRIPT_DIR/cal_acc.py" \
  --benchmark zoombench \
  --judge_json "$SCRIPT_DIR/judge/zoombench/${MODEL_TAG}_mcq_seed42_answer.jsonl" \
  --benchmark_json "$SCRIPT_DIR/zoombench.json"
