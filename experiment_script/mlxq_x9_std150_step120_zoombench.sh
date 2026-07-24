set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
cd "$V"
bash scripts/run_zoombench_canonical.sh /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-virl39k-150step-trial301829143/global_step_120 std_virl39k_step120 0 8302
