set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
cd "$V"
bash scripts/run_zoombench_canonical.sh \
  checkpoints/Vision-OPD-baseline-Qwen3-VL-2B-Instruct-virl39k-filtered-90step-trial301829143/global_step_90 \
  answerhint_2b_virl39k_90step_step90 0 8273
