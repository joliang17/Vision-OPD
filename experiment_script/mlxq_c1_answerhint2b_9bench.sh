set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
BACKEND=vllm_server PORT=8245 \
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/checkpoints/Vision-OPD-baseline-Qwen3-VL-2B-Instruct-trial301683547/global_step_62" \
MODEL_NAME=answerhint_2b_step62_clean_rerun \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench GPU_IDS=0 \
bash shell_scripts/eval_model_temp0_4096.sh
