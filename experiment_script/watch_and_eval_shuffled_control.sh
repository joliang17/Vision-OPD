#!/bin/bash
# Waits for the shuffled_control training process (PID passed as $1) to exit, then merges its
# final checkpoint and launches a multi-GPU VLMEvalKit 7-benchmark eval automatically.
# Designed to survive the launching shell/session disconnecting (run this under nohup+disown).
set -uo pipefail

WAIT_PID="${1:-4110945}"
GPUS="${GPUS:-4,5,6,7}"
CKPT_DIR="checkpoints/Vision-OPD-noimg-shuffled-control-Qwen3-VL-2B-Instruct"
MODEL_NAME="noimg-shuffled-control-step62"
VISION_OPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLMEVAL_ROOT="$(cd "${VISION_OPD_ROOT}/../VLMEvalKit" && pwd)"

echo "$(date -Is) watcher started, waiting for PID ${WAIT_PID} (shuffled_control) to exit"
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
done
echo "$(date -Is) PID ${WAIT_PID} is gone"

cd "$VISION_OPD_ROOT"
STEP=$(cat "${CKPT_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null)
if [[ -z "$STEP" ]]; then
    echo "$(date -Is) ERROR: could not read latest_checkpointed_iteration.txt under ${CKPT_DIR}"
    exit 1
fi
echo "$(date -Is) latest checkpoint step: ${STEP}"

echo "$(date -Is) waiting for GPUs ${GPUS} to go idle"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
for _ in $(seq 1 60); do
    BUSY=0
    for g in "${GPU_ARR[@]}"; do
        MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 999999)
        if [[ "$MEM" -gt 2000 ]]; then
            BUSY=1
        fi
    done
    if [[ "$BUSY" -eq 0 ]]; then
        break
    fi
    sleep 30
done
echo "$(date -Is) GPUs ${GPUS} look idle (or timed out waiting), proceeding"

echo "$(date -Is) merging ${CKPT_DIR}/global_step_${STEP}"
bash scripts/merge_checkpoint.sh "${CKPT_DIR}/global_step_${STEP}"
echo "$(date -Is) merge exited with code $?"

cd "$VLMEVAL_ROOT"
echo "$(date -Is) launching multi-GPU (${GPUS}) VLMEvalKit eval for ${MODEL_NAME}"
MODEL_PATH="${VISION_OPD_ROOT}/${CKPT_DIR}/global_step_${STEP}" \
MODEL_NAME="${MODEL_NAME}" \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K \
GPU_IDS="${GPUS}" \
BACKGROUND=1 \
bash shell_scripts/eval_model_temp0_4096.sh
echo "$(date -Is) eval driver launched (BACKGROUND=1, driver exits immediately; check its own pid=/log= for progress)"
