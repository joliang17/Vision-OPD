set -euo pipefail
source /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/miniconda3/etc/profile.d/conda.sh
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
V=${OPSD_ROOT}/Vision-OPD
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
PATH=$V/scripts/qwen35_shim:$PATH BACKEND=vllm_server PORT=8264 \
REQUEST_TIMEOUT=900 RETRY=4 \
MODEL_PATH=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/Qwen3.5-4B \
MODEL_NAME=vanilla_qwen35_4b_hr8k_rerun \
DATASETS=HRBench8K GPU_IDS=0 \
bash shell_scripts/eval_model_temp0_4096.sh
