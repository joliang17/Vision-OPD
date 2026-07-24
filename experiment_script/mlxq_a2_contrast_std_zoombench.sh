set -euo pipefail
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
cd "$V"
PATH=$V/scripts/qwen35_shim:$PATH bash scripts/run_zoombench_canonical.sh \
  /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3.5-4B-trial301683547/global_step_62 \
  contrast_std_qwen35_4b_step62 0 8244
