set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
cd "$V"
bash scripts/run_zoombench_canonical.sh \
  checkpoints/Vision-OPD-baseline-Qwen3-VL-2B-Instruct-trial301683547/global_step_62 \
  answerhint_2b_step62_clean_rerun 0 8271
