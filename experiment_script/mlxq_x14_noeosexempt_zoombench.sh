set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
cd "$V"
bash scripts/run_zoombench_canonical.sh /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-contrast-noeosexempt-Qwen3-VL-2B-virl39k-90step-trial301761390/global_step_90 noeosexempt_2b_virl39k_step90 0 8308
