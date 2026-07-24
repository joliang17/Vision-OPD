set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
cd "$V"
bash scripts/run_zoombench_canonical.sh \
  checkpoints/Vision-OPD-contrast-standard-uniformweight-Qwen3-VL-2B-virl39k-90step-trial301829143/global_step_90 \
  uniformweight_2b_virl39k_90step_step90 0 8275
