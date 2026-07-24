#!/bin/bash
# Keep launching the OPSD baseline experiment until it exits successfully.
#
# Usage:
#   MODEL_SIZE=2B WANDB_ENABLE=1 bash scripts/run_experiment_baseline_forever.sh trainer.total_epochs=1
#
# Useful knobs:
#   RETRY_SLEEP_SECONDS=300      seconds between failed attempts
#   MAX_ATTEMPTS=0               0 means retry forever
#   WAIT_FOR_GPU_FREE_GIB=0      set >0 to wait for selected GPUs to have this much free memory
#   CUDA_VISIBLE_DEVICES=0,1,2,3 restrict the run to specific GPUs

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASELINE_SCRIPT="${SCRIPT_DIR}/run_experiment_baseline.sh"

RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-300}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
WAIT_FOR_GPU_FREE_GIB="${WAIT_FOR_GPU_FREE_GIB:-0}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/forever}"
mkdir -p "$LOG_DIR"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

print_pip_list() {
    cat <<'EOF'
Missing runtime dependency. Install with:
  python3 -m pip install \
    tensorboard==2.20.0 \
    tensorboard-data-server==0.7.2 \
    absl-py==2.4.0 \
    markdown==3.10.2 \
    -i https://bytedpypi.byted.org/simple

If using WANDB_ENABLE=1, also ensure byted-wandb is installed:
  python3 -m pip install -U byted-wandb -i https://bytedpypi.byted.org/simple
EOF
}

check_python_deps() {
    python3 - <<'PY'
import importlib.util
import sys

missing = []
for module in ("tensorboard",):
    if importlib.util.find_spec(module) is None:
        missing.append(module)

if missing:
    print("Missing Python module(s): " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PY
}

check_gpu_free() {
    python3 - "$WAIT_FOR_GPU_FREE_GIB" "${CUDA_VISIBLE_DEVICES:-}" "${TRAINER_N_GPUS_PER_NODE:-}" <<'PY'
import os
import subprocess
import sys

threshold_gib = float(sys.argv[1])
visible = sys.argv[2].strip()
requested = sys.argv[3].strip()
if threshold_gib <= 0:
    sys.exit(0)

try:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
except Exception as exc:
    print(f"Could not query GPU memory: {exc}", file=sys.stderr)
    sys.exit(1)

free_mib = {}
for line in out.splitlines():
    if not line.strip():
        continue
    idx, free = [part.strip() for part in line.split(",", 1)]
    free_mib[int(idx)] = int(free)

if visible:
    gpus = [int(x) for x in visible.split(",") if x.strip().isdigit()]
else:
    count = int(requested) if requested.isdigit() else len(free_mib)
    gpus = sorted(free_mib)[:count]

threshold_mib = int(threshold_gib * 1024)
not_ready = [(idx, free_mib.get(idx, 0)) for idx in gpus if free_mib.get(idx, 0) < threshold_mib]
if not_ready:
    readable = ", ".join(f"gpu{idx}={free/1024:.1f}GiB" for idx, free in not_ready)
    print(f"Waiting for GPU free memory >= {threshold_gib:.1f}GiB: {readable}", file=sys.stderr)
    sys.exit(1)
PY
}

if [[ ! -x "$BASELINE_SCRIPT" ]]; then
    echo "[$(timestamp)] Missing executable baseline script: $BASELINE_SCRIPT" >&2
    exit 1
fi

attempt=1
while true; do
    if (( MAX_ATTEMPTS > 0 && attempt > MAX_ATTEMPTS )); then
        echo "[$(timestamp)] Reached MAX_ATTEMPTS=$MAX_ATTEMPTS; stopping." >&2
        exit 1
    fi

    echo "[$(timestamp)] Attempt ${attempt} starting."

    if ! check_python_deps; then
        print_pip_list >&2
        exit 1
    fi

    while ! check_gpu_free; do
        sleep "$RETRY_SLEEP_SECONDS"
    done

    log_file="${LOG_DIR}/baseline_attempt_${attempt}_$(date '+%Y%m%d_%H%M%S').log"
    echo "[$(timestamp)] Logging to $log_file"

    (
        cd "$PROJECT_ROOT" || exit 1
        bash "$BASELINE_SCRIPT" "$@"
    ) 2>&1 | tee "$log_file"
    status=${PIPESTATUS[0]}

    if [[ "$status" -eq 0 ]]; then
        echo "[$(timestamp)] Attempt ${attempt} succeeded."
        exit 0
    fi

    echo "[$(timestamp)] Attempt ${attempt} failed with exit code ${status}; retrying in ${RETRY_SLEEP_SECONDS}s." >&2
    attempt=$((attempt + 1))
    sleep "$RETRY_SLEEP_SECONDS"
done
