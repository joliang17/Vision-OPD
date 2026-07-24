set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
BACKEND=vllm_server PORT=8262 \
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/checkpoints/Vision-OPD-contrast-conservative-Qwen3-VL-4B-Instruct-geometry3k/global_step_65" \
MODEL_NAME=geo3k_cons4b_step65 \
DATASETS=WeMath,MathVerse_MINI,MMMU_DEV_VAL,OCRBench,MathVista_MINI,MMStar,HallusionBench GPU_IDS=0 \
bash shell_scripts/eval_model_temp0_4096.sh
