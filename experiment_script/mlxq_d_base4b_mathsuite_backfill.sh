set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
BACKEND=vllm_server PORT=8263 \
MODEL_PATH="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17" \
MODEL_NAME=base_4b_mathsuite \
DATASETS=WeMath,MathVerse_MINI,MMMU_DEV_VAL,OCRBench GPU_IDS=0 \
bash shell_scripts/eval_model_temp0_4096.sh
