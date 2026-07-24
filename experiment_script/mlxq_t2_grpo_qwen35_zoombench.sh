set -euo pipefail
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
cd "$V"
PATH=$V/scripts/qwen35_shim:$PATH bash scripts/run_zoombench_canonical.sh \
  checkpoints/Vision-OPD-grpo-baseline-default-Qwen3.5-4B-trial301761390/global_step_195 \
  grpo_baseline_default_qwen35_4b_step195 0 8030
