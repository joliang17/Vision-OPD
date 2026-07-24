#!/bin/bash
# Hold one CUDA device for interactive debugging without launching training.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash experiment_script/hold_one_device_debug.sh
#
# Optional:
#   HOLD_ALLOC_MB=0    allocate this much GPU memory before sleeping

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HOLD_ALLOC_MB="${HOLD_ALLOC_MB:-0}"

echo "Project root: $PROJECT_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HOLD_ALLOC_MB=$HOLD_ALLOC_MB"
echo "PID: $$"
echo "Press Ctrl-C to stop."

python3 - <<'PY'
import os
import signal
import time

import torch

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
alloc_mb = int(os.environ.get("HOLD_ALLOC_MB", "0"))

print(f"Python PID: {os.getpid()}", flush=True)
print(f"CUDA_VISIBLE_DEVICES={visible}", flush=True)

held = None
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Using logical cuda:0 -> {name}, total={total:.1f} GiB", flush=True)
    torch.empty(1, device="cuda")
    if alloc_mb > 0:
        # bfloat16 is 2 bytes, so this holds approximately alloc_mb MiB.
        numel = alloc_mb * 1024 * 1024 // 2
        held = torch.empty(numel, dtype=torch.bfloat16, device="cuda")
        held.fill_(0)
        print(f"Holding about {alloc_mb} MiB on cuda:0", flush=True)
else:
    print("CUDA is not available; holding CPU process only.", flush=True)

stop = False

def handle_stop(signum, frame):
    global stop
    stop = True

signal.signal(signal.SIGTERM, handle_stop)
signal.signal(signal.SIGINT, handle_stop)

while not stop:
    time.sleep(60)

print("Stopping debug hold.", flush=True)
PY
