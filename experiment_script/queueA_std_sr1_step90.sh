set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-sr1-filtered-90step/global_step_90" \
MODEL_NAME=std_sr1_90step_step90 \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
GPU_IDS=0 \
bash shell_scripts/eval_via_vllm_server.sh
