set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/Vision-OPD"
pip3 install -r "${OPSD_ROOT}/VLMEvalKit/requirements_arnold.txt"
bash scripts/run_zoombench_canonical.sh "/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-reversekl-Qwen3-VL-2B-virl39k-90step-trial301783374/global_step_90" contrast_reversekl_2b_virl39k_step90 0 8266
