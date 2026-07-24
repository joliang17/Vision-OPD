set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
cd "$V"
bash scripts/run_zoombench_canonical.sh /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-baseline-Qwen3-VL-4B-virl39k-UNFILTERED1img-90step-len4096-trial301783374/global_step_90 answerhint_unfiltered_4b_step90 0 8317
