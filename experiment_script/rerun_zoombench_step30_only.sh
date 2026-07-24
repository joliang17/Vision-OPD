set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/Vision-OPD"
bash scripts/run_zoombench_canonical.sh \
  checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_30 \
  contrast_standard_virl39k_90step_step30 0 8010
