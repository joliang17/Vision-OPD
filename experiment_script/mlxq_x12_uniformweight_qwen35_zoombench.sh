set -euo pipefail
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
cd "$V"
PATH=$V/scripts/qwen35_shim:$PATH bash scripts/run_zoombench_canonical.sh \
  ${V}/checkpoints/Vision-OPD-contrast-standard-uniformweight-Qwen3.5-4B-virl39k-90step-trial301829143/global_step_90 \
  uniformweight_qwen35_virl39k_step90 0 8292
