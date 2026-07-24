#!/usr/bin/env bash
# Follow-up driver for the 2026-07-14 task-5 (90-step) retrains.
# Polls (not `wait`, since this runs as its own process tree and can't wait on
# another shell's PIDs) for both conservative 90-step trainings to reach their
# final checkpoint, then merges + prunes them. Does not chain any further
# training/eval automatically -- that's left as an explicit next step.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG="logs/followup_20260714_merge.log"

log() { echo "[$(date)] $*" | tee -a "$LOG"; }

wait_for_step() {
  local run_dir="$1" step="$2" pid="$3"
  log "waiting for ${run_dir} to reach global_step_${step}/actor (tracking pid ${pid})"
  while true; do
    if [ -d "${run_dir}/global_step_${step}/actor" ]; then
      log "${run_dir} reached global_step_${step}"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      log "WARNING: pid ${pid} for ${run_dir} is gone but global_step_${step}/actor never appeared -- run likely failed"
      return 1
    fi
    sleep 60
  done
}

if wait_for_step checkpoints/Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-virl39k-filtered-90step 90 68133; then
  bash scripts/merge_checkpoint.sh checkpoints/Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_90 >> "$LOG" 2>&1
  bash scripts/prune_checkpoints.sh checkpoints/Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-virl39k-filtered-90step 3 >> "$LOG" 2>&1
  log "merged+pruned conservative-virl39k-90step"
fi

if wait_for_step checkpoints/Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-sr1-filtered-90step 90 68134; then
  bash scripts/merge_checkpoint.sh checkpoints/Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-sr1-filtered-90step/global_step_90 >> "$LOG" 2>&1
  bash scripts/prune_checkpoints.sh checkpoints/Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-sr1-filtered-90step 3 >> "$LOG" 2>&1
  log "merged+pruned conservative-sr1-90step"
fi

log "followup driver done -- merge/prune only, eval and further training left as explicit next steps"
