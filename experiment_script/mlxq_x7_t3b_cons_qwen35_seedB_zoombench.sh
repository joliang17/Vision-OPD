set -euo pipefail
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
cd "$V"
PATH=$V/scripts/qwen35_shim:$PATH bash scripts/run_zoombench_canonical.sh \
  ${V}/checkpoints/Vision-OPD-contrast-conservative-Qwen3.5-4B-virl39k-filtered-90step-trial301783374/global_step_90 \
  cons_qwen35_virl39k_step90_seedB 0 8287
