#!/bin/bash
# Backfill HallusionBench for the 20 FCE points that never ran it (10/20/40/50/70/80/100/110/130/140,
# FC1+FC4). All 6-bench data for these already exists in the same generic_eval work_dir; this only
# adds the missing HallusionBench file. GPU_IDS below are picked from `nvidia-smi` at launch time
# (all 8 free except keep_gpu baseline) — 6 lanes in parallel, 2 GPUs held back as spares.
set -uo pipefail
V=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
cd "$V"
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
conda activate qwen35
export PYTHONPATH="$V/.syspkg_shim:/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit"

FC1_DIR=checkpoints/Vision-OPD-contrast-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374
FC4_DIR=checkpoints/Vision-OPD-baseline-Qwen3-VL-2B-virl39k-UNFILTERED1img-150step-keepall-trial301783374
STEPS="10 20 40 50 70 80 100 110 130 140"
GPUS=(0 2 3 5 6 7)

run_one() {
  local run="$1" ckpt_dir="$2" step="$3" gpu="$4"
  local name="${run}_step${step}_server"
  local log="logs/hallu_backfill_${run}_step${step}.log"
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit
  BACKEND=vllm_server \
  MODEL_PATH="$V/${ckpt_dir}/global_step_${step}" \
  MODEL_NAME="${name}" \
  DATASETS=HallusionBench \
  GPU_IDS="${gpu}" \
  bash shell_scripts/eval_model_temp0_4096.sh > "$V/${log}" 2>&1
  echo "[$(date)] ${name} on GPU${gpu} exit=$? log=${log}" >> "$V/logs/hallu_backfill_driver.log"
  cd "$V"
}

jobs_list=()
for s in $STEPS; do jobs_list+=("fc1_uniform_unfiltered ${FC1_DIR} ${s}"); done
for s in $STEPS; do jobs_list+=("fc4_opsd_unfiltered ${FC4_DIR} ${s}"); done

echo "[$(date)] starting HallusionBench backfill, ${#jobs_list[@]} jobs, $(( ${#GPUS[@]} )) parallel lanes" >> logs/hallu_backfill_driver.log

declare -A gpu_pid   # gpu -> running pid (unset/empty = free)
for job in "${jobs_list[@]}"; do
  read -r run ckpt_dir step <<< "$job"
  # find a free GPU, blocking until one opens up
  gpu=""
  while [ -z "$gpu" ]; do
    for g in "${GPUS[@]}"; do
      p="${gpu_pid[$g]:-}"
      if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then
        gpu="$g"
        break
      fi
    done
    [ -z "$gpu" ] && sleep 10
  done
  run_one "$run" "$ckpt_dir" "$step" "$gpu" &
  gpu_pid[$gpu]="$!"
  sleep 5
done
wait
echo "[$(date)] all ${#jobs_list[@]} HallusionBench backfill jobs finished" >> logs/hallu_backfill_driver.log
