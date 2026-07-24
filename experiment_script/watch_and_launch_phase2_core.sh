#!/bin/bash
# Waits for the noimg-4B training process (PID passed as $1, default: the currently running
# Vision-OPD-noimg-Qwen3-VL-4B-Instruct run) to exit and GPU4-7 to go idle, then launches
# Phase 2-核心's two pending experiments sequentially on GPU4-7:
#   1. uniform_x (pure EMA control, no token weighting)
#   2. shuffled_control (real weight distribution, shuffled token<->weight correspondence)
# Designed to survive the launching shell/session disconnecting (run this under nohup+disown).
set -uo pipefail

WAIT_PID="${1:-3066872}"
GPUS="${GPUS:-4,5,6,7}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "$(date -Is) watcher started, waiting for PID ${WAIT_PID} (noimg-4B) to exit"
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
done
echo "$(date -Is) PID ${WAIT_PID} is gone"

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

echo "$(date -Is) launching uniform_x on GPU ${GPUS}"
CUDA_VISIBLE_DEVICES="$GPUS" TRAINER_N_GPUS_PER_NODE=4 bash scripts/run_experiment_noimg_uniform_x.sh
echo "$(date -Is) uniform_x exited with code $?"

echo "$(date -Is) launching shuffled_control on GPU ${GPUS}"
CUDA_VISIBLE_DEVICES="$GPUS" TRAINER_N_GPUS_PER_NODE=4 bash scripts/run_experiment_noimg_shuffled_control.sh
echo "$(date -Is) shuffled_control exited with code $?"

echo "$(date -Is) both Phase 2-核心 runs done"
