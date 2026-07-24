set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
cd "$V"
bash scripts/run_zoombench_canonical.sh \
  checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-4B-virl39k-filtered-90step-trial301761390/global_step_90 \
  contrast_std_4b_virl39k_step90 0 8285
