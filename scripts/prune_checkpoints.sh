#!/usr/bin/env bash
# Prune a COMPLETED training run's checkpoints to the standing retention policy
# (docs/CLAUDE.md "Checkpoint retention"): ~3 evenly-spaced checkpoints for short (~60-step)
# runs, ~5 for long (400+) runs, always keeping the final step. Dense save_freq=10 saving stays
# ON during training (it's what makes crash resume cheap — 2026-07-10's quota crash lost <1 step
# thanks to it); this script is the after-completion cleanup that keeps the disk quota alive.
#
# Usage: bash scripts/prune_checkpoints.sh <checkpoint_run_dir> [num_to_keep]
#   e.g. bash scripts/prune_checkpoints.sh checkpoints/Vision-OPD-foo-2B  3
# Keeps: the final step (from latest_checkpointed_iteration.txt or max step present) plus
# (num_to_keep - 1) evenly spaced earlier steps chosen from what exists. Prints the plan and
# asks for confirmation unless FORCE=1.
set -euo pipefail

RUN_DIR="${1:?usage: prune_checkpoints.sh <run_dir> [num_to_keep]}"
KEEP="${2:-3}"

steps=$(ls "${RUN_DIR}" | grep -oE 'global_step_[0-9]+' | sed 's/global_step_//' | sort -n | uniq)
[ -z "${steps}" ] && { echo "no checkpoints under ${RUN_DIR}"; exit 1; }
final=$(echo "${steps}" | tail -1)

mapfile -t arr <<< "${steps}"
n=${#arr[@]}
declare -A keep_set
keep_set[$final]=1
if [ "${KEEP}" -gt 1 ] && [ "$n" -gt 1 ]; then
  for i in $(seq 1 $((KEEP - 1))); do
    idx=$(( (i * (n - 1)) / KEEP ))
    keep_set[${arr[$idx]}]=1
  done
fi

echo "Run: ${RUN_DIR} (found ${n} checkpoints, final=${final})"
echo "Keeping: ${!keep_set[@]}"
to_delete=()
for s in "${arr[@]}"; do
  [ -z "${keep_set[$s]:-}" ] && to_delete+=("${RUN_DIR}/global_step_${s}")
done
[ ${#to_delete[@]} -eq 0 ] && { echo "nothing to delete"; exit 0; }
printf 'Deleting %d checkpoints:\n' "${#to_delete[@]}"
printf '  %s\n' "${to_delete[@]}"

if [ "${FORCE:-0}" != "1" ]; then
  read -r -p "proceed? [y/N] " ans
  [ "${ans}" != "y" ] && { echo "aborted"; exit 1; }
fi
rm -rf "${to_delete[@]}"
echo "pruned."
